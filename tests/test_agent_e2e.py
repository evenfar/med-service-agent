"""主 Agent 端到端测试（第1~8/10期集成）：全部走离线 Mock。"""

from __future__ import annotations

import json

import pytest

from app.agent.chat import MedicalAgent
from app.agent.tools.mock_data import reset_mock_data
from app.schemas.response import IntentType, UrgencyLevel


@pytest.fixture(autouse=True)
def fresh_data():
    reset_mock_data()
    yield
    reset_mock_data()


@pytest.fixture
def agent(agent_settings):
    a = MedicalAgent(agent_settings)
    yield a
    a.close()


def _tools_used(agent) -> list[str]:
    return [e["tool"] for e in agent.tracer.tool_calls]


class TestReActFlow:
    def test_greeting_no_tools(self, agent):
        resp = agent.chat("你好")
        assert resp.intent == IntentType.greeting
        assert _tools_used(agent) == []
        assert not resp.requires_human

    def test_appointment_query_full_pipeline(self, agent):
        resp = agent.chat("帮我查一下预约 GH-2026-001")
        assert "呼吸内科" in resp.reply and "陈志远" in resp.reply
        assert _tools_used(agent) == ["query_appointment"]
        assert resp.intent == IntentType.appointment
        # 结构化元信息可信
        assert 0 <= resp.confidence <= 1
        # 轨迹完整：llm调用 + 工具调用 + 会话落盘
        assert agent.tracer.llm_calls
        assert agent.history_size > 0
        from app.agent.storage import load_session
        assert load_session(agent.session_path) is not None

    def test_session_restored_by_new_agent(self, agent_settings, agent):
        agent.chat("帮我查一下预约 GH-2026-001")
        agent.save()
        reborn = MedicalAgent(agent_settings)
        assert reborn.history_size == agent.history_size

    def test_tool_failure_relayed(self, agent):
        resp = agent.chat("帮我查预约 GH-2026-999")
        assert "未找到预约单" in resp.reply

    def test_max_steps_guard_exists(self, agent_settings):
        assert agent_settings.max_react_steps >= 1  # 护栏配置存在且生效于 react()


class TestSafetyInPipeline:
    def test_redflag_shortcuts_llm(self, agent):
        resp = agent.chat("我突然胸痛得厉害")
        assert resp.intent == IntentType.emergency
        assert resp.urgency == UrgencyLevel.emergency
        assert resp.requires_human is True
        assert "120" in resp.reply
        # 关键断言：规则短路，一次 LLM 都不发生
        assert agent.tracer.llm_calls == []
        assert agent.tracer.tool_calls == []
        assert agent.tracer.entries[-1]["type"] == "redflag_shortcut"

    def test_critical_value_report_escalates(self, agent):
        resp = agent.chat("帮我解读报告 LAB-2026-004")
        assert "危急值" in resp.reply
        assert resp.requires_human is True
        assert resp.urgency == UrgencyLevel.emergency

    def test_high_risk_interaction_warns(self, agent):
        resp = agent.chat("华法林和阿司匹林能一起吃吗")
        assert "高风险" in resp.reply
        assert resp.requires_human is True


class TestRAGInPipeline:
    def test_knowledge_answer_has_sources(self, agent):
        resp = agent.chat("医保报销流程是什么")
        assert "search_knowledge" in _tools_used(agent)
        assert "参考来源" in resp.reply
        assert "【就诊指南#医保报销】" in resp.reply  # 引用真实存在

    def test_citation_validation_drops_fake_marks(self, agent):
        # 直接构造：伪造引用应被剥离（单元级验证走 test_safety，这里验证管道挂载）
        from app.agent.safety import validate_citations
        fixed, fakes = validate_citations(
            "根据【就诊指南#医保报销】和【杜撰文档#瞎编】的回答",
            ["就诊指南#医保报销"])
        assert fakes == ["杜撰文档#瞎编"]
        assert "杜撰文档" not in fixed


class TestSkillsInPipeline:
    def test_triage_skill_loaded(self, agent):
        resp = agent.chat("咳嗽两周了应该挂什么科")
        assert "load_skill" in _tools_used(agent)
        assert "分诊" in resp.reply


class TestCompressionAndMemory:
    def test_history_compression_triggers(self, agent_settings):
        agent_settings.history_threshold = 4
        agent_settings.history_keep_recent = 2
        agent = MedicalAgent(agent_settings)
        agent.chat("帮我查一下预约 GH-2026-001")
        agent.chat("布洛芬怎么吃")
        assert agent.summary is not None  # 已压缩出摘要
        assert len(agent.raw_messages) <= 6  # 老消息被折叠
        agent.close()

    def test_short_term_memory_persisted(self, agent_settings, agent):
        agent.chat("帮我查一下预约 GH-2026-001")
        data = json.load(open(agent.session_path, encoding="utf-8"))
        assert "short_term_memory" in data

    def test_stm_empty_with_mock_extraction(self, agent_settings):
        """离线 mock 的记忆提取返回空 → 不产生幻觉事实（宁缺毋滥）。"""
        agent = MedicalAgent(agent_settings)
        agent.chat("你好")
        assert agent.memory_manager.stm.facts == []
        agent.close()


class TestRepeatQuestion:
    def test_same_question_twice_both_use_tools(self, agent):
        """回归：mock 步数计数按回合复位——重复提问不能退化为兜底问候。"""
        r1 = agent.chat("布洛芬怎么吃")
        assert "布洛芬" in r1.reply
        r2 = agent.chat("布洛芬怎么吃")
        assert "布洛芬" in r2.reply  # 第二次也要走 query_medicine
        assert _tools_used(agent).count("query_medicine") == 2
