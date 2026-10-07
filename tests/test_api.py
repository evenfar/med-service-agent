"""API 服务层测试：全离线（monkeypatch BASE_SETTINGS 为 tmp 路径的 mock 配置）。

覆盖：/health、/chat 两轮会话（工具调用 + 急症红线短路）、/chat/stream SSE 协议、
/reset 会话数不增、非法 session_id 拒绝。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app import api


@pytest.fixture
def client(tmp_path, shared_index, monkeypatch):
    """离线 TestClient：BASE_SETTINGS 换成 tmp 路径，lifespan 退出时关闭全部 agent。"""
    monkeypatch.setattr(api, "BASE_SETTINGS",
                        _settings(tmp_path, shared_index))
    with TestClient(api.app) as c:
        yield c


def _settings(tmp_path, kb_index: str) -> Settings:
    return Settings(
        mock_mode=True,
        kb_index_path=kb_index,
        session_path=str(tmp_path / "session.json"),
        memory_dir=str(tmp_path / "memory"),
    )


def _get(client: TestClient, url: str, **kw):
    r = client.get(url, **kw)
    assert r.status_code == 200, r.text
    return r


def test_health(client):
    data = _get(client, "/health").json()
    assert data["status"] == "ok"
    assert isinstance(data["offline"], bool)
    assert data["sessions"] == 0


def test_chat_two_rounds(client):
    sid = "test-chat-01"
    # 第1轮：预约查询 → 工具调用轨迹 + 领域内容
    r = client.post("/chat", json={"session_id": sid,
                                   "message": "帮我查一下预约 GH-2026-001"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "呼吸内科" in d["reply"]
    assert d["intent"] == "appointment"
    assert d["events"], "本轮应有增量事件"
    types = {e["type"] for e in d["events"]}
    assert "tool_call" in types or "llm_call" in types

    # 第2轮：急症红线 → 规则短路（零LLM，requires_human 强制为真）
    r2 = client.post("/chat", json={"session_id": sid,
                                    "message": "我突然胸痛"})
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["requires_human"] is True
    assert d2["urgency"] == "emergency"
    assert "120" in d2["reply"]
    types2 = {e["type"] for e in d2["events"]}
    assert types2 == {"redflag_shortcut"}   # 零LLM：只有红线短路事件
    # 同一 session_id 复用 agent：会话数仍为 1
    assert _get(client, "/health").json()["sessions"] == 1


def test_chat_stream_sse(client):
    r = _get(client, "/chat/stream",
             params={"session_id": "test-stream-01",
                     "message": "帮我查一下预约 GH-2026-001", "mode": "mock"})
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "tool_call" in body            # 思考过程事件已推送
    assert "event: done" in body          # 以 done 收尾
    assert body.rstrip().endswith("}")    # done 携带完整 MedicalResponse JSON
    # done 的 payload 里能解析出结构化字段
    tail = body[body.rindex("event: done"):]
    payload = tail[tail.index("data: ") + 6:].strip()
    done = json.loads(payload)
    assert done["intent"] == "appointment"
    assert "呼吸内科" in done["reply"]


def test_reset_keeps_session_count(client):
    sid = "test-reset-01"
    r = client.post("/chat", json={"session_id": sid, "message": "你好"})
    assert r.status_code == 200
    before = _get(client, "/health").json()["sessions"]
    rr = client.post("/reset", json={"session_id": sid})
    assert rr.status_code == 200
    assert _get(client, "/health").json()["sessions"] == before  # 重建而非叠加


def test_invalid_session_id_rejected(client):
    r = client.post("/chat", json={"session_id": "../evil", "message": "你好"})
    assert r.status_code == 400
