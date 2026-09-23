"""LLM 层测试：重试退避 + Mock 客户端行为（第1期/第3期）。"""

from __future__ import annotations

import pytest

from app.agent.tracer import Tracer
from app.config.settings import Settings
from app.llm.client import (LLMError, MockLLMClient, TransientLLMError,
                            with_retry, _is_transient)
from app.schemas.response import MedicalResponse, UrgencyLevel


# ---------------- with_retry ----------------

class TestRetry:
    def test_retries_transient_then_succeeds(self):
        calls = {"n": 0}
        delays = []

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientLLMError("429")
            return "ok"

        out = with_retry(flaky, max_retries=3, base_delay=1.0,
                         sleeper=delays.append)
        assert out == "ok"
        assert calls["n"] == 3
        assert delays == [1.0, 2.0]  # 指数退避 1s, 2s

    def test_non_transient_raises_immediately(self):
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            raise ValueError("400 参数错误")

        with pytest.raises(ValueError):
            with_retry(bad, max_retries=3, sleeper=lambda _: None)
        assert calls["n"] == 1  # 4xx 不重试

    def test_exhausted_raises_llm_error(self):
        def always_429():
            raise TransientLLMError("429")
        with pytest.raises(LLMError):
            with_retry(always_429, max_retries=2, sleeper=lambda _: None)

    def test_is_transient_by_status_code(self):
        class FakeErr(Exception):
            def __init__(self, code):
                super().__init__(str(code))
                self.status_code = code

        assert _is_transient(FakeErr(429))
        assert _is_transient(FakeErr(503))
        assert not _is_transient(FakeErr(400))
        assert not _is_transient(FakeErr(401))


# ---------------- MockLLMClient ----------------

class TestMockClient:
    def _mk(self):
        return MockLLMClient(tracer=Tracer())

    def test_router_classification(self):
        client = self._mk()
        assert client.chat([{"role": "user", "content": "布洛芬怎么吃"}],
                           purpose="router").content == "medication"
        assert client.chat([{"role": "user", "content": "看报告 LAB-2026-001"}],
                           purpose="router").content == "report"
        assert client.chat([{"role": "user", "content": "预约挂号"}],
                           purpose="router").content == "appointment"
        assert client.chat([{"role": "user", "content": "随便聊聊"}],
                           purpose="router").content == "appointment"  # 默认兜底

    def test_react_two_step_with_tool_result_echo(self):
        client = self._mk()
        sys_msg = {"role": "system", "content": "s1"}
        user = {"role": "user", "content": "帮我查一下预约 GH-2026-001"}
        tools = [{"type": "function", "function": {"name": "query_appointment"}}]

        r1 = client.chat([sys_msg, user], tools=tools, purpose="react")
        assert r1.tool_calls[0]["name"] == "query_appointment"
        assert r1.tool_calls[0]["arguments"]["appointment_id"] == "GH-2026-001"

        # 模拟工具结果回填后，第二次调用应产出 final 并引用工具字段
        history = [sys_msg, user,
                   {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "query_appointment",
                                                 "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "call_1",
                    "content": '{"success": true, "appointment": {'
                               '"department": "呼吸内科", "doctor": "陈志远",'
                               '"date": "2026-09-25", "time": "09:30",'
                               '"status": "booked"}}'}]
        r2 = client.chat(history, tools=tools, purpose="react")
        assert not r2.tool_calls
        assert "呼吸内科" in r2.content and "陈志远" in r2.content

    def test_react_greeting_no_tools(self):
        client = self._mk()
        r = client.chat([{"role": "system", "content": "s"},
                         {"role": "user", "content": "你好"}],
                        tools=[{"type": "function",
                                "function": {"name": "x"}}], purpose="react")
        assert not r.tool_calls and r.content

    def test_react_purpose_prefix_for_subagents(self):
        client = self._mk()
        r = client.chat([{"role": "system", "content": "s"},
                         {"role": "user", "content": "布洛芬怎么吃"}],
                        tools=[{"type": "function",
                                "function": {"name": "query_medicine"}}],
                        purpose="react:medication")
        assert r.tool_calls and r.tool_calls[0]["name"] == "query_medicine"

    def test_parse_structured_medical_response(self):
        client = self._mk()
        parsed = client.parse_structured(
            [{"role": "user", "content": "根据查询结果：您预约了呼吸内科…"}],
            MedicalResponse)
        assert parsed.intent.value == "appointment"
        assert parsed.reply.startswith("根据查询结果")
        assert 0 <= parsed.confidence <= 1

    def test_parse_structured_emergency_detection(self):
        client = self._mk()
        parsed = client.parse_structured(
            [{"role": "user", "content": "请立即拨打120前往急诊"}], MedicalResponse)
        assert parsed.urgency == UrgencyLevel.emergency
        assert parsed.requires_human is True

    def test_parse_structured_empty_facts_schema(self):
        from app.agent.memory.models import ShortTermFacts
        client = self._mk()
        parsed = client.parse_structured([{"role": "user", "content": "x"}],
                                         ShortTermFacts)
        assert parsed.facts == []


def test_offline_factory_builds_mock():
    from app.llm.client import build_client, MockLLMClient
    assert isinstance(build_client(Settings(mock_mode=True), Tracer()),
                      MockLLMClient)
