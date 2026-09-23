"""Multi-Agent 测试（第6期）：路由 / 白名单隔离 / 编排端到端。"""

from __future__ import annotations

import pytest

from app.agent.tools.mock_data import reset_mock_data
from app.agent.tools.registry import ToolDeps, build_tool_registry
from app.llm.client import MockLLMClient
from app.multi_agent.agents import SUBAGENT_SPECS
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.router import Router
from app.schemas.response import IntentType, UrgencyLevel


@pytest.fixture(autouse=True)
def fresh_data():
    reset_mock_data()
    yield
    reset_mock_data()


class TestRouter:
    @pytest.fixture
    def router(self):
        return Router(MockLLMClient())

    def test_routes(self, router):
        assert router.route("帮我查预约 GH-2026-001") == "appointment"
        assert router.route("布洛芬怎么吃") == "medication"
        assert router.route("看看报告 LAB-2026-001") == "report"
        assert router.route("胸痛") == "emergency"

    def test_fallback_default(self, router):
        assert router.route("今天天气不错") == "appointment"


class TestSpecs:
    def test_four_subagents(self):
        assert set(SUBAGENT_SPECS) == {"appointment", "medication",
                                       "report", "emergency"}

    def test_tool_whitelists_minimal(self):
        assert "cancel_appointment" not in SUBAGENT_SPECS["medication"].tools
        assert "query_medicine" not in SUBAGENT_SPECS["appointment"].tools
        assert len(SUBAGENT_SPECS["emergency"].tools) == 0  # 急症零工具

    def test_whitelist_actually_filters(self):
        full = build_tool_registry(ToolDeps())
        med = full.subset(set(SUBAGENT_SPECS["medication"].tools))
        assert "query_appointment" not in med.names()


class TestOrchestratorE2E:
    @pytest.fixture
    def orch(self, agent_settings):
        agent_settings.multi_agent_enabled = True
        o = MultiAgentOrchestrator(agent_settings)
        yield o
        o.close()

    def _route(self, orch) -> str:
        routes = [e["agent"] for e in orch.tracer.entries if e["type"] == "route"]
        return routes[-1] if routes else ""

    def test_routes_appointment(self, orch):
        resp = orch.chat("帮我查一下预约 GH-2026-001")
        assert self._route(orch) == "挂号分诊助理"
        assert "呼吸内科" in resp.reply

    def test_routes_medication(self, orch):
        resp = orch.chat("华法林和阿司匹林能一起吃吗")
        assert self._route(orch) == "用药咨询助理"
        assert "高风险" in resp.reply

    def test_routes_report(self, orch):
        resp = orch.chat("帮我看看报告 LAB-2026-004")
        assert self._route(orch) == "报告解读助理"
        assert "危急值" in resp.reply

    def test_redflag_before_router(self, orch):
        """急症在路由之前就被规则短路 —— 优先级验证。"""
        resp = orch.chat("我突然胸痛得厉害")
        assert resp.urgency == UrgencyLevel.emergency
        assert self._route(orch) == ""  # 没走到路由
        assert orch.tracer.llm_calls == []

    def test_subagent_only_uses_whitelisted_tools(self, orch):
        orch.chat("布洛芬怎么吃")
        tools = {e["tool"] for e in orch.tracer.tool_calls}
        assert tools <= set(SUBAGENT_SPECS["medication"].tools)
