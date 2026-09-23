"""Multi-Agent 编排器（第6期）：路由 → 子 Agent（ReAct）→ 结构化提取。

对比 ecom 的改进：ecom 的编排器复制了单 Agent 的全部会话/压缩/提取代码；
本项目继承 BaseAgentRuntime，chat() 中间段只有"路由 + 委托子 Agent"，
公共骨架（红旗短路/安全后处理/记忆/压缩/持久化）全部复用基类。
"""

from __future__ import annotations

from typing import Optional

from app.agent.base import BaseAgentRuntime
from app.agent.memory.manager import MemoryManager
from app.agent.rag.retriever import build_retriever
from app.agent.skills.loader import SkillManager
from app.agent.tools.registry import ToolDeps, ToolRegistry, build_tool_registry
from app.agent.tracer import Tracer
from app.config.settings import Settings
from app.multi_agent.agents import SUBAGENT_SPECS
from app.multi_agent.router import Router
from app.schemas.response import MedicalResponse


class MultiAgentOrchestrator(BaseAgentRuntime):
    """多 Agent 模式：Router 分流到 挂号/用药/报告/急诊 四个子 Agent。"""

    def __init__(self, settings: Optional[Settings] = None,
                 session_path: Optional[str] = None,
                 tracer: Optional[Tracer] = None, client=None):
        super().__init__(settings or Settings(), session_path, tracer, client)

    def _init_components(self) -> None:
        s = self.settings
        self.router = Router(self.client)
        if s.memory_enabled:
            self.memory_manager = MemoryManager(
                client=self.client, user_id=s.memory_user_id,
                memory_dir=s.memory_dir, enabled=True,
                max_ltm_facts=s.max_ltm_facts)
        self.skill_manager = SkillManager(s.skills_dir, s.skills_enabled)
        retriever = build_retriever(s, self.tracer)
        # 全量注册一次，各子 Agent 用 subset 白名单取视图
        self._full_registry = build_tool_registry(ToolDeps(
            tracer=self.tracer, retriever=retriever,
            memory_manager=self.memory_manager,
            skill_manager=self.skill_manager))

    def _restore_extra(self, loaded: dict) -> None:
        if self.memory_manager and loaded.get("short_term_memory"):
            self.memory_manager.restore_stm(loaded["short_term_memory"])

    # ---- 主流程 ----

    def chat(self, user_input: str) -> MedicalResponse:
        emergency = self.handle_redflag(user_input)
        if emergency is not None:
            return self.finish_turn(emergency)

        self.raw_messages.append({"role": "user", "content": user_input})
        agent_key = self.router.route(user_input, self.raw_messages[:-1])
        spec = SUBAGENT_SPECS[agent_key]
        self.tracer.log_route(spec.name)
        print(f"\n🔀 [路由] → {spec.name}")

        system = spec.prompt
        if self.skill_manager and self.skill_manager.enabled:
            system += self.skill_manager.build_catalog_prompt()
        registry = (self._full_registry.subset(set(spec.tools))
                    if spec.tools else None)

        final, used, outputs = self.react(system, registry,
                                          purpose=f"react:{agent_key}")
        final, forced_human = self.safety_post_process(final, used, outputs)

        result = self.extract_structured(final)
        if forced_human:
            result.requires_human = True
        return self.finish_turn(result)

    def close(self) -> None:
        super().close()
        from app.agent.tools.mcp_bridge import close_mcp
        close_mcp(self._full_registry)
