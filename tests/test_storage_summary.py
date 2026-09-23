"""会话持久化与历史摘要测试（第2期）。"""

from __future__ import annotations

import json

from app.agent.storage import delete_session, load_session, save_session
from app.agent.summarizer import render_transcript, summarize
from app.llm.client import MockLLMClient


class TestStorage:
    def test_roundtrip_with_tool_messages(self, tmp_path):
        path = str(tmp_path / "session.json")
        messages = [
            {"role": "user", "content": "帮我查预约 GH-2026-001"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "query_appointment",
                                          "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            {"role": "assistant", "content": "已查到"},
        ]
        save_session(path, messages, "此前摘要", {"facts": ["用户问过预约"]})
        loaded = load_session(path)
        assert loaded["messages"] == messages
        assert loaded["summary"] == "此前摘要"
        assert loaded["short_term_memory"]["facts"] == ["用户问过预约"]

    def test_corrupt_file_degrades_to_new_session(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        assert load_session(path) is None

    def test_version_mismatch_rejected(self, tmp_path):
        path = str(tmp_path / "old.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 99, "messages": []}, f)
        assert load_session(path) is None

    def test_missing_file(self, tmp_path):
        assert load_session(str(tmp_path / "none.json")) is None

    def test_delete(self, tmp_path):
        path = str(tmp_path / "s.json")
        save_session(path, [], None)
        delete_session(path)
        delete_session(path)  # 幂等
        assert load_session(path) is None

    def test_no_tmp_left_after_save(self, tmp_path):
        path = str(tmp_path / "s.json")
        save_session(path, [{"role": "user", "content": "hi"}], None)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []


class TestSummarizer:
    def test_render_transcript_includes_tools(self):
        msgs = [
            {"role": "user", "content": "查预约"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "query_appointment",
                                          "arguments": "{\"a\":1}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "结果" * 300},
        ]
        text = render_transcript(msgs, max_tool_chars=50)
        assert "用户：查预约" in text
        assert "query_appointment" in text
        assert len([l for l in text.splitlines() if l.startswith("工具结果")][0]) < 80

    def test_summarize_via_mock(self):
        client = MockLLMClient()
        out = summarize(client, [{"role": "user", "content": "我之前查过血常规报告"}],
                        prev_summary=None)
        assert out.startswith("（摘要）")
