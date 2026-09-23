"""工具层测试（第3期）：注册表协议 + 各工具行为 + 白名单。"""

from __future__ import annotations

import json

import pytest

from app.agent.tools import mock_data
from app.agent.tools.appointment import cancel_appointment, query_appointment
from app.agent.tools.department import query_department
from app.agent.tools.lab import query_lab_report
from app.agent.tools.medicine import check_drug_interaction, query_medicine
from app.agent.tools.mock_data import reset_mock_data
from app.agent.tools.registry import ToolDeps, ToolRegistry, build_tool_registry


@pytest.fixture(autouse=True)
def fresh_data():
    reset_mock_data()
    yield
    reset_mock_data()


@pytest.fixture
def registry():
    return build_tool_registry(ToolDeps())


# ---------------- 注册表协议 ----------------

class TestRegistry:
    EXPECTED = {"query_appointment", "cancel_appointment", "query_medicine",
                "check_drug_interaction", "query_lab_report",
                "query_department", "search_knowledge",
                "recall_user_memory", "load_skill"}

    def test_all_tools_registered(self, registry):
        assert set(registry.names()) == self.EXPECTED

    def test_definitions_openai_format(self, registry):
        for d in registry.definitions:
            assert d["type"] == "function"
            fn = d["function"]
            assert fn["name"] and fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_execute_unknown_tool(self, registry):
        out = json.loads(registry.execute("hack_tool", {}))
        assert "error" in out and "query_appointment" in out["error"]

    def test_execute_missing_required_param(self, registry):
        out = json.loads(registry.execute("query_appointment", {}))
        assert "必填参数" in out["error"]

    def test_execute_wrong_type(self, registry):
        out = json.loads(registry.execute("query_appointment",
                                          {"appointment_id": 123}))
        assert "类型" in out["error"]

    def test_tool_exception_becomes_observation(self, registry):
        # cancel_appointment 缺 action 会正常返回错误 dict；这里注入一个真抛异常的工具
        reg = ToolRegistry()
        reg.register("boom", "x", {"type": "object", "properties": {}},
                     lambda **kw: (_ for _ in ()).throw(RuntimeError("炸了")))
        out = json.loads(reg.execute("boom", {}))
        assert "RuntimeError" in out["error"]

    def test_subset_whitelist(self, registry):
        view = registry.subset({"query_medicine", "check_drug_interaction"})
        assert set(view.names()) == {"query_medicine", "check_drug_interaction"}
        # 白名单视图调用不到被排除的工具
        out = json.loads(view.execute("query_appointment", {"appointment_id": "GH-2026-001"}))
        assert "未知工具" in out["error"]
        # 全量注册表不受影响
        assert registry.has("query_appointment")


# ---------------- 工具行为 ----------------

class TestAppointmentTools:
    def test_query_found(self):
        out = query_appointment("GH-2026-001")
        assert out["success"] and out["appointment"]["department"] == "呼吸内科"

    def test_query_not_found(self):
        out = query_appointment("GH-2026-999")
        assert not out["success"] and "未找到" in out["error"]

    def test_cancel_mutates_state(self):
        out = cancel_appointment("GH-2026-001", action="cancel")
        assert out["success"]
        assert mock_data.APPOINTMENTS["GH-2026-001"]["status"] == "cancelled"
        again = cancel_appointment("GH-2026-001", action="cancel")
        assert not again["success"]  # 二次取消被拒

    def test_reschedule_requires_new_date(self):
        out = cancel_appointment("GH-2026-005", action="reschedule")
        assert not out["success"]
        ok = cancel_appointment("GH-2026-005", action="reschedule",
                                new_date="2026-10-08")
        assert ok["success"] and "2026-10-08" in ok["message"]


class TestMedicineTools:
    def test_query_medicine(self):
        out = query_medicine("布洛芬")
        assert out["success"] and out["medicine"]["type"] == "OTC"

    def test_query_medicine_unknown(self):
        out = query_medicine("仙丹")
        assert not out["success"]

    def test_interaction_high_risk(self):
        out = check_drug_interaction("阿司匹林", "华法林")
        assert out["severity"] == "高风险" and "出血" in out["effect"]

    def test_interaction_uncovered_is_explicit(self):
        out = check_drug_interaction("氯雷他定", "阿莫西林")
        assert out["success"] and out["severity"] == "未收录"
        assert "不等于无风险" in out["advice"]  # 不知道 ≠ 安全

    def test_interaction_unknown_drug(self):
        out = check_drug_interaction("仙丹", "布洛芬")
        assert not out["success"]


class TestLabTool:
    def test_normal_report(self):
        out = query_lab_report("LAB-2026-001")
        assert out["success"] and out["has_critical"] is False

    def test_critical_report_flagged(self):
        out = query_lab_report("LAB-2026-004")
        assert out["has_critical"] is True


class TestDepartmentTool:
    def test_match_by_department_name(self):
        out = query_department("帮我预约心血管内科的号")
        assert out["success"] and out["department"]["name"] == "心血管内科"

    def test_match_by_symptom(self):
        out = query_department("反复咳嗽咳痰")
        assert out["success"] and out["department"]["name"] == "呼吸内科"

    def test_no_match_is_explicit(self):
        out = query_department("心情不好想找人聊天")
        assert not out["success"]


class TestKnowledgeToolNoRetriever:
    def test_disabled_retriever_returns_error(self):
        from app.agent.tools.knowledge import search_knowledge
        out = search_knowledge(None, "医保")
        assert not out["success"] and "未启用" in out["error"]
