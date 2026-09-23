"""Agent 运行时基类（第6期重构产物）：会话/压缩/ReAct/结构化提取/安全后处理。

对比 ecom 的改进：ecom 把同样约200行代码复制粘贴在 EcomAgent 与
MultiAgentOrchestrator 两处（提取/降级/压缩/会话骨架），改一处漏一处；
本项目把"一次对话回合的骨架"全部抽到这里，单Agent与编排器只负责各自的中间段，
ReAct 循环全系统只有这一份实现（SubAgent 也复用它）。
"""

from __future__ import annotations

import json
from typing import Optional

from app.agent.memory.manager import MemoryManager
from app.agent.safety import (append_sources_if_missing, detect_red_flags,
                              emergency_reply, scan_output, validate_citations)
from app.agent.storage import delete_session, load_session, save_session
from app.agent.summarizer import summarize
from app.agent.tracer import Tracer
from app.agent.tools.registry import ToolRegistry
from app.config.settings import Settings
from app.llm.client import build_client
from app.schemas.response import IntentType, MedicalResponse, UrgencyLevel

EXTRACT_SYSTEM = ("基于以下健康助手回复内容提取结构化信息。"
                  "reply 字段直接使用原文，不要修改或缩减。"
                  "急症相关内容（急诊/120/危急值）必须把 urgency 标为 emergency "
                  "且 requires_human 为 true。")


class BaseAgentRuntime:
    """会话 + ReAct + 结构化提取 + 安全后处理的公共运行时。"""

    def __init__(self, settings: Settings, session_path: Optional[str] = None,
                 tracer: Optional[Tracer] = None, client=None):
        self.settings = settings
        self.tracer = tracer or Tracer()
        self.client = client or build_client(settings, self.tracer)
        self.session_path = session_path or settings.session_path
        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        self.memory_manager: Optional[MemoryManager] = None
        self._init_components()          # 子类钩子：装配 memory/skills/tools
        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            self._restore_extra(loaded)

    # 子类钩子 ------------------------------------------------------

    def _init_components(self) -> None:
        """子类在这里构建 memory/tools/skills 等组件。"""

    def _restore_extra(self, loaded: dict) -> None:
        """会话恢复钩子（如恢复短期记忆）。"""

    # ---------- 会话 ----------

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def save(self) -> None:
        save_session(self.session_path, self.raw_messages, self.summary,
                     short_term_memory=(self.memory_manager.stm_to_dict()
                                        if self.memory_manager else None))

    def reset_session(self) -> None:
        self.raw_messages = []
        self.summary = None
        if self.memory_manager:
            self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    # ---------- 消息组装 ----------

    def render_messages(self, system_content: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system_content}]
        if self.memory_manager:
            messages.extend(self.memory_manager.build_prompt_sections())
        if self.summary:
            messages.append({"role": "system",
                             "content": f"此前对话摘要：\n{self.summary}"})
        messages.extend(self.raw_messages)
        return messages

    # ---------- ReAct 循环（全系统唯一实现） ----------

    def react(self, system_content: str, registry: Optional[ToolRegistry],
              purpose: str = "react",
              max_steps: Optional[int] = None
              ) -> tuple[str, list[str], list[tuple[str, str]]]:
        """返回 (最终文本, 本轮调用过的工具名, [(工具名, 原始输出JSON)])。"""
        steps = max_steps or self.settings.max_react_steps
        used: list[str] = []
        outputs: list[tuple[str, str]] = []
        tools = registry.definitions if registry else None
        for _ in range(steps):
            resp = self.client.chat(self.render_messages(system_content),
                                    tools=tools, purpose=purpose)
            if not resp.tool_calls:
                self.raw_messages.append({"role": "assistant", "content": resp.content})
                return resp.content, used, outputs
            self.raw_messages.append({
                "role": "assistant", "content": resp.content,
                "tool_calls": [{"id": tc["id"], "type": "function",
                                "function": {"name": tc["name"],
                                             "arguments": _dump_args(tc["arguments"])}}
                               for tc in resp.tool_calls]})
            for tc in resp.tool_calls:
                out = (registry.execute(tc["name"], tc["arguments"])
                       if registry else '{"error": "当前没有可用工具"}')
                used.append(tc["name"])
                outputs.append((tc["name"], out))
                self.raw_messages.append({"role": "tool", "tool_call_id": tc["id"],
                                          "content": out})
        # 步数用尽：去掉工具再给一次收尾机会（防"截断在半路"）
        resp = self.client.chat(self.render_messages(system_content), purpose=purpose)
        self.raw_messages.append({"role": "assistant", "content": resp.content})
        return resp.content, used, outputs

    # ---------- 结构化提取 ----------

    def extract_structured(self, text: str) -> MedicalResponse:
        return self.client.parse_structured(
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": text}],
            MedicalResponse, purpose="extract")

    # ---------- 安全（第10期，单Agent与多Agent共用） ----------

    def handle_redflag(self, user_input: str) -> Optional[MedicalResponse]:
        """急症红线短路：规则命中直接返回急诊应答，不进 LLM（零延迟零漏报）。"""
        flags = detect_red_flags(user_input)
        if not flags:
            return None
        self.tracer.log("redflag_shortcut", flags=flags)
        self.raw_messages.append({"role": "user", "content": user_input})
        return MedicalResponse(
            intent=IntentType.emergency, confidence=1.0,
            reply=emergency_reply(flags), requires_human=True,
            urgency=UrgencyLevel.emergency)

    def safety_post_process(self, final: str, used: list[str],
                            outputs: list[tuple[str, str]]) -> tuple[str, bool]:
        """回复出站前的三道处理：注入扫描 → 引用校验 → 危急值升级。"""
        forced_human = False
        if "search_knowledge" in used:
            safe, cleaned = scan_output(final)
            if not safe:
                self.tracer.log("injection_blocked", original_len=len(final))
                final, forced_human = cleaned, True
            sources: list[str] = []
            for name, out in outputs:
                if name != "search_knowledge":
                    continue
                try:
                    data = json.loads(out)
                    sources += [c["source"] for c in data.get("chunks", [])
                                if isinstance(c, dict)]
                except (json.JSONDecodeError, AttributeError):
                    pass
            final, fakes = validate_citations(final, sources)
            if fakes:
                self.tracer.log("fake_citations_removed", fakes=fakes)
            final = append_sources_if_missing(final, sources)
        for name, out in outputs:
            if name != "query_lab_report":
                continue
            try:  # 按字段取值而非子串匹配——序列化格式变化不会让防线失效
                if json.loads(out).get("has_critical") is True:
                    final = ("该报告含危急值，请立即联系开单医生或前往急诊！\n" + final)
                    forced_human = True
            except json.JSONDecodeError:
                pass
        return final, forced_human

    # ---------- 历史压缩 ----------

    def compress_history(self) -> None:
        split = len(self.raw_messages) - self.settings.history_keep_recent
        while split > 0 and self.raw_messages[split].get("role") == "tool":
            split -= 1  # 不从 tool 消息中间切开（孤儿 tool 消息会破坏协议）
        if split <= 0:
            return
        old, recent = self.raw_messages[:split], self.raw_messages[split:]
        self.summary = summarize(self.client, old, self.summary)
        self.raw_messages = recent
        print(f"\n💾 [历史压缩] {len(old)} 条老消息 → summary({len(self.summary)}字)\n")

    def maybe_compress(self) -> None:
        if len(self.raw_messages) > self.settings.history_threshold:
            self.compress_history()

    # ---------- 回合收尾 ----------

    def finish_turn(self, result: MedicalResponse) -> MedicalResponse:
        self.raw_messages.append(
            {"role": "assistant",
             "content": result.model_dump_json(ensure_ascii=False)})
        if self.memory_manager:
            self.memory_manager.update_short_term(self.raw_messages[-6:])
        self.maybe_compress()
        self.save()
        return result

    def close(self) -> None:
        """会话结束：巩固长期记忆。子类追加自己的资源清理。"""
        if self.memory_manager:
            self.memory_manager.consolidate_to_long_term(
                self.raw_messages, self.summary)


def _dump_args(args: dict) -> str:
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"
