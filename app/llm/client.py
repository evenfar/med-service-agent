"""LLM 抽象层：统一接口 + 重试退避 + 离线 Mock。

三个工程要点（对比 ecom 的改进）：
1. 重试退避是客户端能力而非调用方的事：只有瞬态错误（429/5xx/超时/网络）重试，
   4xx 参数错误立刻失败 —— 与 multi-agent 项目同款策略。
2. Mock 客户端与真实客户端同一接口、会读取对话历史（含工具结果），
   因此离线跑出来的轨迹与真实模式结构一致，测试/评估/演示都可信。
3. tracer 注入：每次调用（含用途 purpose）都计量 token 与延迟，运行时可观测。
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from app.agent.tracer import Tracer
from app.config.settings import Settings, is_offline


class LLMError(RuntimeError):
    """LLM 调用最终失败（重试耗尽或不可重试错误）。"""


class TransientLLMError(LLMError):
    """可瞬态恢复的错误（429/5xx/超时/网络），值得重试。"""


# ---------------- 重试：可独立单测的纯函数 ----------------

def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, TransientLLMError):
        return True
    try:
        import openai
        transient_types = (openai.RateLimitError, openai.APITimeoutError,
                           openai.APIConnectionError, openai.InternalServerError)
        if isinstance(exc, transient_types):
            return True
    except ImportError:
        pass
    status = getattr(exc, "status_code", None)
    return status in (408, 409, 429, 500, 502, 503, 504)


def with_retry(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    is_transient: Callable[[Exception], bool] = _is_transient,
) -> Any:
    """指数退避重试：只重试瞬态错误，其余立刻抛出。sleeper 可注入便于测试。"""
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 —— 分类后决定重试或抛出
            if not is_transient(e):
                raise
            last = e
            if attempt < max_retries - 1:
                sleeper(base_delay * (2 ** attempt))
    raise LLMError(f"LLM请求重试{max_retries}次仍失败: {last}")


# ---------------- 统一接口与响应对象 ----------------

class LLMResponse:
    def __init__(self, content: str = "", tool_calls: Optional[list[dict]] = None,
                 prompt_tokens: int = 0, completion_tokens: int = 0):
        self.content = content
        # [{"id": str, "name": str, "arguments": dict}]
        self.tool_calls = tool_calls or []
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None,
             purpose: str = "react") -> LLMResponse:
        """对话补全。tools 为 OpenAI function calling 格式；purpose 仅用于计量。"""

    @abstractmethod
    def parse_structured(self, messages: list[dict], schema: Type[BaseModel],
                         purpose: str = "extract") -> BaseModel:
        """结构化输出：优先用 response_format，失败降级为 prompt 引导 JSON。"""


# ---------------- 真实客户端（OpenAI 兼容） ----------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


class OpenAICompatClient(BaseLLMClient):
    def __init__(self, settings: Settings, tracer: Tracer, inner_client=None):
        self._s = settings
        self._tracer = tracer
        if inner_client is not None:
            self._inner = inner_client
        else:
            import openai
            self._inner = openai.OpenAI(api_key=settings.openai_api_key,
                                        base_url=settings.openai_base_url)

    # ---- 内部：带重试的原始调用 ----

    def _call(self, make: Callable[[], Any]) -> Any:
        def guarded():
            try:
                return make()
            except Exception as e:  # noqa: BLE001
                if _is_transient(e):
                    raise TransientLLMError(str(e)) from e
                raise
        return with_retry(guarded, max_retries=self._s.llm_max_retries,
                          base_delay=self._s.llm_retry_base_delay)

    def chat(self, messages, tools=None, temperature=None, max_tokens=None,
             purpose="react") -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._s.model_name,
            "messages": messages,
            "temperature": self._s.temperature if temperature is None else temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        t0 = time.perf_counter()
        resp = self._call(lambda: self._inner.chat.completions.create(**kwargs))
        self._trace(purpose, resp, (time.perf_counter() - t0) * 1000)
        msg = resp.choices[0].message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=msg.content or "", tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def parse_structured(self, messages, schema, purpose="extract"):
        t0 = time.perf_counter()
        try:
            resp = self._call(lambda: self._inner.beta.chat.completions.create(
                model=self._s.model_name, messages=messages,
                temperature=0.0, response_format=schema))
            self._trace(purpose, resp, (time.perf_counter() - t0) * 1000)
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                raise LLMError("parsed 为空")
            return parsed
        except Exception:
            return self._parse_fallback(messages, schema, purpose)

    def _parse_fallback(self, messages, schema, purpose):
        """response_format 不被接口支持时，用 prompt 引导 JSON 再本地校验。"""
        t0 = time.perf_counter()
        sys_prompt = (
            "从下面的文本中提取结构化信息，只输出一个 JSON 对象（禁止代码块），"
            "必须符合此 JSON Schema：\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            + "\n文本："
        )
        merged = [{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": messages[-1]["content"]}]
        resp = self._call(lambda: self._inner.chat.completions.create(
            model=self._s.model_name, messages=merged, temperature=0.0))
        self._trace(purpose, resp, (time.perf_counter() - t0) * 1000)
        try:
            return schema.model_validate_json(_strip_fences(resp.choices[0].message.content or ""))
        except ValidationError as e:
            raise LLMError(f"结构化输出解析失败: {e}") from e

    def _trace(self, purpose: str, resp, latency_ms: float) -> None:
        usage = getattr(resp, "usage", None)
        self._tracer.log_llm(
            purpose=purpose, model=getattr(resp, "model", "") or self._s.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms)


# ---------------- 离线 Mock：脚本化但会读对话历史 ----------------

class MockLLMClient(BaseLLMClient):
    """与真实客户端同接口。行为由 purpose + 用户消息内容决定，
    且 final 文本从历史里的工具结果中提取字段 —— 轨迹结构与真实模式一致。"""

    def __init__(self, tracer: Optional[Tracer] = None):
        self._tracer = tracer
        self._counts: dict[tuple, int] = {}

    # ---- 计数：同一(系统提示,当前问题)按步数推进 ----

    def _bump(self, key: tuple) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    @staticmethod
    def _last_user(messages) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    @staticmethod
    def _tool_results(messages, since_last_user: bool = False) -> list[tuple[str, dict]]:
        """(工具名, 结果dict)，按调用顺序。

        since_last_user=True 时只取"最近一条用户消息之后"的部分 ——
        多轮会话里历史轮次的工具结果不属于当前问题的答案素材。
        """
        start = 0
        if since_last_user:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    start = i
                    break
        id2name: dict[str, str] = {}
        results: list[tuple[str, dict]] = []
        for m in messages[start:]:
            role = m.get("role")
            if role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) if "function" in tc else tc
                    id2name[tc.get("id", "")] = fn.get("name", "")
            elif role == "tool":
                name = id2name.get(m.get("tool_call_id", ""), "?")
                try:
                    results.append((name, json.loads(m.get("content", "{}"))))
                except json.JSONDecodeError:
                    results.append((name, {}))
        return results

    # ---- 接口实现 ----

    def chat(self, messages, tools=None, temperature=None, max_tokens=None,
             purpose="react") -> LLMResponse:
        user = self._last_user(messages)
        if purpose == "router":
            content = self._route(user)
            return self._mk(content, messages)
        if purpose == "summarize":
            content = "（摘要）" + user[:150]
            return self._mk(content, messages)
        if purpose != "react" and not purpose.startswith("react:"):
            return self._mk(f"（mock {purpose}）", messages)
        return self._react(messages, user, bool(tools))

    def parse_structured(self, messages, schema, purpose="extract"):
        text = messages[-1]["content"] if messages else ""
        fields = schema.model_fields
        if "intent" in fields and "reply" in fields:          # MedicalResponse
            from app.schemas.response import IntentType, UrgencyLevel
            return schema(**self._medical_fields(text))
        if "interaction_summary" in fields:                    # 长期记忆提取
            return schema(facts=[], interaction_summary="")
        if "facts" in fields:                                  # 短期记忆提取
            return schema(facts=[])
        try:
            return schema()
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"mock 无法构造 {schema.__name__}: {e}") from e

    def _mk(self, content: str, messages, tool_calls=None) -> LLMResponse:
        if self._tracer:
            approx_prompt = len(json.dumps(messages, ensure_ascii=False)) // 4
            self._tracer.log_llm(purpose="react", model="mock", prompt_tokens=approx_prompt,
                                 completion_tokens=len(content) // 2, latency_ms=1.0)
        return LLMResponse(content=content, tool_calls=tool_calls or [],
                           prompt_tokens=len(content) // 2, completion_tokens=len(content) // 2)

    # ---- ReAct 脚本 ----

    def _react(self, messages, user: str, has_tools: bool) -> LLMResponse:
        key = (hash(messages[0]["content"]) % 10**8, user[:120])
        n = self._bump(key)
        plan = self._plan(user)
        if has_tools and n <= len(plan):
            name, args = plan[n - 1]
            return self._mk("", messages, tool_calls=[
                {"id": f"call_{n}", "name": name, "arguments": args}])
        return self._mk(self._final_text(user, messages), messages)

    @staticmethod
    def _plan(user: str) -> list[tuple[str, dict]]:
        """按用户问题决定工具调用序列（可多步）。"""
        if re.search(r"(你好|您好|hi|hello|在吗)", user, re.IGNORECASE):
            return []
        apt = re.search(r"GH-\d{4}-\d{3}", user)
        if apt:
            plan = [("query_appointment", {"appointment_id": apt.group(0)})]
            if "取消" in user or "退" in user:
                plan.append(("cancel_appointment",
                             {"appointment_id": apt.group(0), "action": "cancel"}))
            return plan
        drugs = [d for d in ("布洛芬", "阿莫西林", "氯雷他定", "二甲双胍",
                             "华法林", "阿司匹林", "硝酸甘油", "对乙酰氨基酚") if d in user]
        if len(drugs) >= 2 or (drugs and any(w in user for w in ("一起", "同时", "联用", "合用", "相互作用"))):
            return [("check_drug_interaction",
                     {"drug_a": drugs[0], "drug_b": drugs[1] if len(drugs) > 1 else "阿司匹林"})]
        if drugs:
            return [("query_medicine", {"name": drugs[0]})]
        if "药" in user:
            return [("query_medicine", {"name": "布洛芬"})]
        # 政策类问句先于"报告"关键词：如"这个报告要注意什么"是追问注意事项而非查报告
        if any(w in user for w in ("医保", "报销", "流程", "怎么", "注意", "须知", "准备", "政策")):
            return [("search_knowledge", {"query": user[:40]})]
        if re.search(r"LAB-\d{4}-\d{3}", user) or \
                any(w in user for w in ("报告", "化验", "血常规", "指标", "检验")):
            m = re.search(r"LAB-\d{4}-\d{3}", user)
            return [("query_lab_report", {"report_id": m.group(0) if m else "LAB-2026-001"})]
        if any(w in user for w in ("分诊", "挂什么科", "推荐科室", "应该看", "哪个科室")):
            return [("load_skill", {"skill_name": "triage"})]
        if any(w in user for w in ("科室", "挂号", "预约")):
            return [("query_department", {"keyword": user[:20]})]
        return []

    def _final_text(self, user: str, messages) -> str:
        results = self._tool_results(messages, since_last_user=True)
        pieces: list[str] = []
        for name, data in results:
            if data.get("success") is False and data.get("error"):
                pieces.append(data["error"])  # 工具失败必须如实转达
                continue
            if name == "query_appointment" and data.get("success"):
                a = data["appointment"]
                pieces.append(f"您预约了{a['department']} {a['doctor']}医生，"
                              f"时间 {a['date']} {a['time']}，当前状态：{a['status']}")
            elif name == "cancel_appointment":
                pieces.append(data.get("message", "预约已处理"))
            elif name == "query_medicine" and data.get("success"):
                m = data["medicine"]
                pieces.append(f"{m['name']}：{m['indication']}；用法：{m['dosage']}；"
                              f"注意：{m['precautions']}")
            elif name == "check_drug_interaction":
                sev = data.get("severity", "")
                pair = f"{data.get('drug_a', '')}与{data.get('drug_b', '')}"
                pieces.append(f"联用{pair}风险等级：{sev}。{data.get('advice', '')}")
            elif name == "query_lab_report" and data.get("success"):
                rep = data["report"]
                abn = [f"{i['name']}({i['value']}{i['unit']}, {i['flag']})"
                       for i in rep["items"] if i.get("flag") in ("偏高", "偏低", "危急")]
                pieces.append(f"报告 {rep['report_id']}（{rep['name']}，{rep['date']}）"
                              f"异常指标：{'、'.join(abn) if abn else '无'}，"
                              f"结论：{rep['conclusion']}")
            elif name == "query_department" and data.get("success"):
                d = data["department"]
                pieces.append(f"建议就诊科室：{d['name']}（{d['description']}）")
            elif name == "search_knowledge":
                for ch in data.get("chunks", [])[:2]:
                    body = ch["text"].split("】", 1)[-1]  # 去掉【doc#section】标签
                    pieces.append(f"根据{ch['source']}：{body[:80]}")
            elif name == "load_skill" and data.get("success"):
                pieces.append("已按分诊技能流程处理：请先说明症状持续时间与严重程度，"
                              "出现高热不退、呼吸困难等情形请立即急诊")
            elif name == "recall_user_memory":
                facts = data.get("long_term", [])
                if facts:
                    pieces.append("结合您的健康档案：" + "；".join(
                        f"[{f['category']}]{f['content']}" for f in facts[:3]))
        if pieces:
            return "根据查询结果：" + "；".join(pieces) + \
                "。以上信息仅供参考，不构成诊疗建议，具体请以医生意见为准。"
        return "您好，我是健康助手小医。请描述您的症状或问题（如挂号、用药、报告解读），" \
               "紧急情况请立即线下就医或呼叫急救电话。"

    # ---- 其他用途的脚本 ----

    @staticmethod
    def _route(user: str) -> str:
        if any(w in user for w in ("胸痛", "呼吸困难", "急救", "大出血", "晕倒", "抽搐")):
            return "emergency"
        if any(w in user for w in ("报告", "化验", "指标", "检验", "血常规")):
            return "report"
        if any(w in user for w in ("药", "服用", "用药", "布洛芬", "阿莫西林", "华法林",
                                    "氯雷他定", "二甲双胍", "阿司匹林", "硝酸甘油",
                                    "对乙酰氨基酚")):
            return "medication"
        return "appointment"

    @staticmethod
    def _medical_fields(text: str) -> dict:
        from app.schemas.response import IntentType, UrgencyLevel
        drug_names = ("布洛芬", "阿莫西林", "氯雷他定", "二甲双胍",
                      "华法林", "阿司匹林", "硝酸甘油", "对乙酰氨基酚")
        if any(w in text for w in ("急诊", "120", "立即就医")):
            intent, urgency = IntentType.emergency, UrgencyLevel.emergency
        elif "您好" in text[:6]:
            intent, urgency = IntentType.greeting, UrgencyLevel.routine
        elif "指标" in text or "报告" in text or "化验" in text:
            intent, urgency = IntentType.lab_report, UrgencyLevel.attention
        elif any(d in text for d in drug_names):
            intent, urgency = IntentType.medication, UrgencyLevel.routine
        elif "预约" in text or "挂号" in text or "科室" in text:
            intent, urgency = IntentType.appointment, UrgencyLevel.routine
        elif "医保" in text or "流程" in text or text.startswith("根据") and "科" in text:
            intent, urgency = IntentType.policy, UrgencyLevel.routine
        elif "您好" in text[:10]:
            intent, urgency = IntentType.greeting, UrgencyLevel.routine
        else:
            intent, urgency = IntentType.other, UrgencyLevel.routine
        requires = any(w in text for w in ("急诊", "120", "转人工", "危急", "高风险"))
        if any(w in text for w in ("危急", "高风险", "异常")) and urgency.value == "routine":
            urgency = UrgencyLevel.attention
        return {"intent": intent, "confidence": 0.85, "reply": text,
                "requires_human": requires, "urgency": urgency,
                "follow_up_question": None}


def build_client(settings: Settings, tracer: Tracer) -> BaseLLMClient:
    """工厂：离线判定集中在一处。"""
    if is_offline(settings):
        return MockLLMClient(tracer=tracer)
    return OpenAICompatClient(settings, tracer)
