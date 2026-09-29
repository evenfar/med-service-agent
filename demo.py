"""教学用 Agent 单文件 Demo：从用户输入走到工具调用和最终回答。

运行：python demo.py
默认输入：请查询预约单 GH-2026-001

这个文件只演示主干流程：
main -> Agent.chat -> ReAct -> 模型提出 tool_call -> ToolRegistry.execute
-> Python 工具 -> 工具结果回填 -> 模型最终答复 -> 安全/结构化/收尾。

FakeLLMClient 是脚本化模型，不会联网、不读取 .env、不消耗 API。
它模拟真实 API 的可见协议，便于逐步读懂消息和工具调用。

真实项目对应关系见每个函数注释。RAG/MCP/Memory/Skill/Multi-Agent/评估
在文件末尾列出扩展挂载点；它们没有在本教学版里重复实现。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 0. 示例配置、示例数据和返回类型
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """教学版配置。真实项目由 app/config/settings.py 从 .env 读取。"""

    model_name: str = "FAKE-MODEL（仅模拟）"
    mock_mode: bool = True
    max_react_steps: int = 4
    session_path: str = "（教学版不写磁盘）"


@dataclass
class LLMResponse:
    """把模型响应统一成两种可能：普通文字，或一个/多个工具调用。"""

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


@dataclass
class MedicalResponse:
    """最终给 CLI 展示的结构化结果。真实类型在 app/schemas/response.py。"""

    intent: str
    confidence: float
    reply: str
    requires_human: bool = False
    urgency: str = "routine"
    follow_up_question: str | None = None


# 查询工具读取这份教学数据；它不是医院数据库。
APPOINTMENTS = {
    "GH-2026-001": {
        "appointment_id": "GH-2026-001",
        "department": "呼吸内科",
        "doctor": "陈志远",
        "date": "2026-09-25",
        "time": "09:30",
        "status": "booked",
        "fee": 50.0,
    }
}


# ---------------------------------------------------------------------------
# 1. 可观测记录：教学版直接打印每个阶段
# ---------------------------------------------------------------------------

class Tracer:
    """真实项目的 app/agent/tracer.py 会记录 LLM、工具和路由统计。"""

    def __init__(self) -> None:
        self.model_calls = 0
        self.tool_calls = 0

    def show(self, event: str, payload: Any) -> None:
        print(f"\n🔎 [TRACE] {event}")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    def log_model_call(self) -> None:
        self.model_calls += 1

    def log_tool_call(self) -> None:
        self.tool_calls += 1

    def summary(self) -> None:
        print("\n── 教学版运行摘要（模拟计数，不是 API 账单）──")
        print(f"模型请求：{self.model_calls} 次 | 工具执行：{self.tool_calls} 次")


# ---------------------------------------------------------------------------
# 2. 工具函数 + 注册表
# ---------------------------------------------------------------------------

def query_appointment(appointment_id: str) -> dict:
    """真实项目：app/agent/tools/appointment.py::query_appointment。"""

    appointment = APPOINTMENTS.get(appointment_id)
    if appointment is None:
        return {"success": False, "error": f"未找到预约单 {appointment_id}"}
    return {"success": True, "appointment": appointment}


@dataclass
class Tool:
    """一项工具包含：名字、说明、参数 Schema、实际 Python 函数。"""

    name: str
    description: str
    parameters: dict
    function: Callable[..., dict]

    def model_definition(self) -> dict:
        """发给模型的定义不包含 Python 函数本身，只包含名称/说明/Schema。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """按模型返回的工具名，查找并执行本地 Python 函数。

    真实项目：app/agent/tools/registry.py::ToolRegistry。
    重要：模型只提出调用请求；真正执行发生在这里。
    """

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.tools = {
            "query_appointment": Tool(
                name="query_appointment",
                description="按预约单号查询预约详情",
                parameters={
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "string"},
                    },
                    "required": ["appointment_id"],
                },
                function=query_appointment,
            )
        }

    def definitions(self) -> list[dict]:
        """Agent 每次请求模型时，把可用工具定义放进 API 请求。"""

        return [tool.model_definition() for tool in self.tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """校验模型参数、调用函数、把结果编码成 JSON 字符串。"""

        started = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            result = {"success": False, "error": f"未知工具：{name}"}
        else:
            required = tool.parameters.get("required", [])
            missing = [key for key in required if key not in arguments]
            if missing:
                result = {"success": False, "error": f"缺少参数：{missing}"}
            else:
                # 只把 Schema 中声明过的字段传给 Python 函数。
                allowed = tool.parameters["properties"]
                safe_arguments = {k: v for k, v in arguments.items() if k in allowed}
                type_check = {
                    "string": lambda value: isinstance(value, str),
                    "integer": lambda value: isinstance(value, int)
                    and not isinstance(value, bool),
                    "number": lambda value: isinstance(value, (int, float))
                    and not isinstance(value, bool),
                    "boolean": lambda value: isinstance(value, bool),
                }
                bad_types = [
                    key for key, value in safe_arguments.items()
                    if allowed[key].get("type") in type_check
                    and not type_check[allowed[key]["type"]](value)
                ]
                if bad_types:
                    result = {"success": False, "error": f"参数类型不正确：{bad_types}"}
                else:
                    try:
                        result = tool.function(**safe_arguments)
                    except Exception as exc:  # 工具失败转成可回填的结果
                        result = {"success": False, "error": f"工具异常：{exc}"}

        self.tracer.log_tool_call()
        self.tracer.show("工具执行完成", {
            "name": name,
            "arguments": arguments,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "result": result,
        })
        return json.dumps(result, ensure_ascii=False)


def build_tool_registry(tracer: Tracer) -> ToolRegistry:
    """真实项目：app/agent/tools/registry.py::build_tool_registry。"""

    registry = ToolRegistry(tracer)
    tracer.show("工具注册完成", {
        "names": list(registry.tools),
        "note": "注册只让工具可供模型选择；还没有执行工具函数。",
    })
    return registry


# ---------------------------------------------------------------------------
# 3. 模拟模型客户端：仅模拟 API 的“请求/响应协议”
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """对应真实项目 app/llm/client.py::OpenAICompatClient。

    第一轮看到预约单号时，模拟模型返回 tool_call。
    下一轮看到 role=tool 的查询结果时，模拟模型生成最终答复。
    """

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.tracer.log_model_call()
        self.tracer.show("模型请求（模拟）", {
            "messages": messages,
            "tools": tools,
        })

        # 找到本轮最新的 user 消息，只看它之后的 tool 结果。
        last_user_index = max(
            i for i, message in enumerate(messages) if message["role"] == "user"
        )
        current_turn = messages[last_user_index + 1:]
        tool_result = next(
            (m for m in reversed(current_turn) if m["role"] == "tool"), None
        )

        if tool_result is not None:
            data = json.loads(tool_result["content"])
            if data.get("success"):
                a = data["appointment"]
                reply = (
                    f"预约单 {a['appointment_id']} 查询结果：{a['department']}，"
                    f"{a['doctor']}医生，预约时间为 {a['date']} {a['time']}，"
                    f"当前状态为已预约，费用为 {a['fee']} 元。"
                )
            else:
                reply = data.get("error", "查询失败")
            response = LLMResponse(content=reply)
        else:
            user_text = messages[last_user_index]["content"]
            match = re.search(r"GH-\d{4}-\d{3}", user_text)
            if match:
                # 这只是模拟模型“选择工具”。这一步没有运行 query_appointment。
                response = LLMResponse(tool_calls=[{
                    "id": "demo-call-001",
                    "name": "query_appointment",
                    "arguments": {"appointment_id": match.group(0)},
                }])
            else:
                response = LLMResponse(content="教学模型只演示预约单号查询。")

        self.tracer.show("模型可见响应（模拟）", asdict(response))
        return response


def build_client(settings: Settings, tracer: Tracer) -> FakeLLMClient:
    """真实项目会根据 mock_mode / API Key 返回 Mock 或 OpenAICompatClient。"""

    # 教学版固定使用 FakeLLMClient，防止误调用真实 API 或消耗额度。
    return FakeLLMClient(tracer)


def load_session(session_path: str) -> dict:
    """真实项目：app/agent/storage.py::load_session。

    为了每次演示都从零开始，这个简化实现固定返回空会话。
    """

    return {"messages": [], "summary": None}


def save_session(session_path: str, messages: list[dict], summary: str | None) -> dict:
    """真实项目：app/agent/storage.py::save_session；教学版不写磁盘。"""

    return {
        "persisted": False,
        "session_path": session_path,
        "message_count": len(messages),
        "summary_present": bool(summary),
    }


# ---------------------------------------------------------------------------
# 4. 公共运行时：历史、ReAct 循环、后处理、结构化和收尾
# ---------------------------------------------------------------------------

class BaseAgentRuntime:
    """真实项目对应 app/agent/base.py::BaseAgentRuntime。"""

    SYSTEM_PROMPT = "你是医疗健康助手；需要预约信息时先调用预约查询工具。"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tracer = Tracer()
        self.client = build_client(settings, self.tracer)
        self.raw_messages: list[dict] = []
        self.summary: str | None = None

        # 和真实项目一样，父类 init 会通过动态派发调用子类的装配钩子。
        self._init_components()
        loaded = load_session(self.settings.session_path)
        self.raw_messages = loaded["messages"]
        self.summary = loaded["summary"]
        self._restore_extra(loaded)
        self.tracer.show("Agent 初始化完成", {
            "model": settings.model_name,
            "registered_tools": list(self.tools.tools),
            "restored_message_count": len(self.raw_messages),
        })

    def _init_components(self) -> None:
        """子类钩子：真实项目在 MedicalAgent 里装配记忆、RAG、技能和工具。"""

    def _restore_extra(self, loaded: dict) -> None:
        """子类钩子：真实项目可在这里恢复短期记忆等状态。"""

    def render_messages(self) -> list[dict]:
        """真实项目还会拼入 Memory、摘要；此处保留 system + 对话历史。"""

        return ([{"role": "system", "content": self.SYSTEM_PROMPT}]
                + self.raw_messages)

    def react(self) -> tuple[str, list[str], list[tuple[str, str]]]:
        """循环：请求模型 -> 执行工具 -> 回填观察结果 -> 再请求模型。"""

        used_tools: list[str] = []
        tool_outputs: list[tuple[str, str]] = []

        for step in range(self.settings.max_react_steps):
            messages = self.render_messages()
            self.tracer.show("ReAct 循环开始", {"step": step + 1})
            response = self.client.chat(messages, self.tools.definitions())

            if not response.tool_calls:
                # 没有 tool_calls 表示模型这轮给了普通答复，ReAct 结束。
                self.raw_messages.append({
                    "role": "assistant", "content": response.content
                })
                return response.content, used_tools, tool_outputs

            # 把 assistant 的工具请求原样放入上下文，包含 call id。
            self.raw_messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })

            for call in response.tool_calls:
                name = call["name"]
                arguments = call["arguments"]
                self.tracer.show("Agent 准备执行工具", {
                    "name": name, "arguments": arguments,
                })
                output = self.tools.execute(name, arguments)
                used_tools.append(name)
                tool_outputs.append((name, output))

                # role=tool 把 Python 函数结果交还给模型；下一次循环会带上它。
                self.raw_messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output,
                })

        return "达到最大工具循环次数，请稍后重试。", used_tools, tool_outputs


class MedicalAgent(BaseAgentRuntime):
    """简化的单 Agent。真实项目还在 chat() 里装配 Skills/RAG/Memory/MCP。"""

    RED_FLAGS = ("胸痛", "呼吸困难", "意识不清", "大出血")

    def _init_components(self) -> None:
        """真实项目对应 MedicalAgent._init_components。"""

        self.tools = build_tool_registry(self.tracer)

    def handle_redflag(self, user_input: str) -> MedicalResponse | None:
        """红旗命中就短路，不进入普通模型/工具循环。"""

        matched = [word for word in self.RED_FLAGS if word in user_input]
        self.tracer.show("红旗检查", {"matched": matched})
        if not matched:
            return None
        return MedicalResponse(
            intent="emergency", confidence=1.0,
            reply="检测到急症信号，请立即拨打 120 或前往急诊。",
            requires_human=True, urgency="emergency",
        )

    def chat(self, user_input: str) -> MedicalResponse:
        """真实项目：app/agent/chat.py::MedicalAgent.chat。"""

        emergency = self.handle_redflag(user_input)
        if emergency is not None:
            return self.finish_turn(emergency)

        self.raw_messages.append({"role": "user", "content": user_input})
        final_text, used, outputs = self.react()
        final_text, forced_human = self.safety_post_process(final_text, used, outputs)
        response = self.extract_structured(final_text)
        if forced_human:
            response.requires_human = True
        return self.finish_turn(response)

    def safety_post_process(
        self, text: str, used: list[str], outputs: list[tuple[str, str]]
    ) -> tuple[str, bool]:
        """真实项目还检查注入、引用和报告危急值；此处演示处理节点。"""

        forced_human = False
        self.tracer.show("安全后处理", {
            "used_tools": used,
            "tool_output_count": len(outputs),
            "forced_human": forced_human,
        })
        return text, forced_human

    def extract_structured(self, text: str) -> MedicalResponse:
        """真实项目调用 client.parse_structured；这里用规则模拟提取结果。"""

        is_appointment = "预约" in text or "挂号" in text
        result = MedicalResponse(
            intent="appointment" if is_appointment else "other",
            confidence=0.99 if is_appointment else 0.7,
            reply=text,
        )
        self.tracer.show("结构化提取结果（模拟）", asdict(result))
        return result

    def finish_turn(self, response: MedicalResponse) -> MedicalResponse:
        """真实项目还更新短期记忆、按阈值压缩历史并写入 session.json。"""

        self.raw_messages.append({
            "role": "assistant",
            "content": json.dumps(asdict(response), ensure_ascii=False),
        })
        save_result = save_session(
            self.settings.session_path, self.raw_messages, self.summary
        )
        self.tracer.show("会话保存位置（教学版只模拟）", save_result)
        return response

    def reset_session(self) -> None:
        self.raw_messages.clear()
        self.summary = None
        print("已清空教学版对话历史。")


# ---------------------------------------------------------------------------
# 5. CLI：创建 Agent、读取输入、展示答复
# ---------------------------------------------------------------------------

def load_settings() -> Settings:
    """真实项目直接构造 Settings()，由 Pydantic 从 .env 加载配置。"""

    return Settings()


def main() -> None:
    settings = load_settings()
    agent = MedicalAgent(settings)
    print("教学版单 Agent（FakeLLMClient：不会调用真实 API）")
    print("输入预约问题；reset 清空历史；quit 退出。")

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "reset":
            agent.reset_session()
            continue

        response = agent.chat(user_input)
        print(f"\n🩺 小医: {response.reply}")
        print(
            f"[意图: {response.intent} | 置信度: {response.confidence:.0%}"
            f" | 紧急度: {response.urgency} | 转人工: {response.requires_human}]"
        )

    agent.tracer.summary()
    print("退出教学版。")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 6. 真实项目其余模块如何挂到这条主链（这里只标位置，不重复实现）
# ---------------------------------------------------------------------------
# Prompt / Response schema : app/prompts/consultant.py, app/schemas/response.py
# 多轮持久化 / 摘要压缩 : app/agent/storage.py, app/agent/summarizer.py
# RAG 知识检索工具       : app/agent/tools/knowledge.py, app/agent/rag/
# Memory 上下文/更新      : app/agent/memory/manager.py
# Skill 目录/按需加载     : app/agent/skills/loader.py + tools/skill_tool.py
# MCP 远程工具桥          : app/agent/tools/mcp_bridge.py
# Multi-Agent 路由        : app/multi_agent/router.py, orchestrator.py
# 安全规则                : app/agent/safety/
# 离线评估                : app/evaluation/ 与 app/scripts/run_eval.py
