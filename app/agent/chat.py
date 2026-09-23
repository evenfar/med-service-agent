"""主 Agent（第1~8期叠加 + 第10期安全）：单 Agent 模式的完整实现。

一次 chat() 的完整流水线：
  红旗短路(规则) → ReAct循环(工具/技能) → 安全后处理(注入/引用/危急值)
  → 结构化提取 → 短期记忆更新 → 历史压缩 → 持久化
"""

from __future__ import annotations

from typing import Optional

from app.agent.base import BaseAgentRuntime
from app.agent.memory.manager import MemoryManager
from app.agent.rag.retriever import build_retriever
from app.agent.skills.loader import SkillManager
from app.agent.tools.mcp_bridge import attach_mcp_tools
from app.agent.tools.registry import ToolDeps, ToolRegistry, build_tool_registry
from app.agent.tracer import Tracer
from app.config.settings import Settings
from app.prompts.consultant import SYSTEM_PROMPT
from app.schemas.response import MedicalResponse


class MedicalAgent(BaseAgentRuntime):
    """健康咨询助手「小医」（单 Agent 模式）。"""

    def __init__(self, settings: Optional[Settings] = None,
                 session_path: Optional[str] = None,
                 tracer: Optional[Tracer] = None, client=None):
        super().__init__(settings or Settings(), session_path, tracer, client)

    # ---- 组件装配（在基类 __init__ 的钩子阶段调用，会话恢复之前） ----

    def _init_components(self) -> None:
        s = self.settings
        if s.memory_enabled:
            self.memory_manager = MemoryManager(
                client=self.client, user_id=s.memory_user_id,
                memory_dir=s.memory_dir, enabled=True,
                max_ltm_facts=s.max_ltm_facts)
        self.skill_manager = SkillManager(s.skills_dir, s.skills_enabled)
        self.retriever = build_retriever(s, self.tracer)
        registry = build_tool_registry(ToolDeps(
            tracer=self.tracer, retriever=self.retriever,
            memory_manager=self.memory_manager,
            skill_manager=self.skill_manager))
        if s.mcp_enabled:
            attach_mcp_tools(registry, s, self.tracer)  # 失败自动降级为纯本地
        self.tools: ToolRegistry = registry

    def _restore_extra(self, loaded: dict) -> None:
        if self.memory_manager and loaded.get("short_term_memory"):
            self.memory_manager.restore_stm(loaded["short_term_memory"])

    # ---- 主流程 ----

    def chat(self, user_input: str) -> MedicalResponse:
        emergency = self.handle_redflag(user_input)
        if emergency is not None:
            return self.finish_turn(emergency)

        self.raw_messages.append({"role": "user", "content": user_input})
        system = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            system += self.skill_manager.build_catalog_prompt()

        final, used, outputs = self.react(system, self.tools)
        final, forced_human = self.safety_post_process(final, used, outputs)

        result = self.extract_structured(final)
        if forced_human:
            result.requires_human = True
        return self.finish_turn(result)

    def close(self) -> None:
        super().close()
        from app.agent.tools.mcp_bridge import close_mcp
        close_mcp(self.tools)  # 释放 MCP 后台线程与 HTTP 会话（若启用）
