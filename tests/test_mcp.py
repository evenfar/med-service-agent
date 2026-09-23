"""MCP 测试（第4期）：schema 转换、降级路径、server 工具函数。

真正的 Streamable HTTP 联调（起 server + client 跨进程）放在 docs/第4期
作为手动实验；这里覆盖可离线确定的部分。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.tools.registry import ToolDeps, ToolRegistry, build_tool_registry
from app.config.settings import Settings


class TestConverter:
    def _tool(self):
        return SimpleNamespace(
            name="query_appointment",
            description="查询预约",
            inputSchema={"type": "object",
                         "properties": {"appointment_id": {"type": "string"}},
                         "required": ["appointment_id"]})

    def test_to_openai_format(self):
        from app.mcp_client.converter import mcp_tool_to_openai
        d = mcp_tool_to_openai(self._tool())
        assert d["type"] == "function"
        assert d["function"]["name"] == "query_appointment"
        assert "appointment_id" in d["function"]["parameters"]["properties"]

    def test_defaults_for_missing_schema(self):
        from app.mcp_client.converter import mcp_tools_to_openai
        bare = SimpleNamespace(name="x", description="y", inputSchema=None)
        out = mcp_tools_to_openai([bare])
        assert out[0]["function"]["parameters"]["type"] == "object"


class TestBridgeDegradation:
    def test_unreachable_server_degrades_to_local(self):
        from app.agent.tools.mcp_bridge import attach_mcp_tools
        settings = Settings(mock_mode=True,
                            mcp_enabled=True,
                            mcp_server_url="http://127.0.0.1:59999/mcp")
        registry = build_tool_registry(ToolDeps())
        ok = attach_mcp_tools(registry, settings)
        assert ok is False
        # 降级后本地工具完好
        assert registry.has("query_appointment")


@pytest.fixture(scope="module")
def med_server():
    pytest.importorskip("mcp")
    from app.agent.rag.retriever import build_index
    build_index(Settings(mock_mode=True))  # server 按默认路径建/读索引
    import mcp_server.server as srv
    return srv


class TestServerFunctions:
    """直接调用 server 模块的工具函数（FastMCP 装饰器返回原函数）。"""

    def test_query_appointment_tool(self, med_server):
        out = json.loads(med_server.query_appointment_tool("GH-2026-001"))
        assert out["success"] and out["appointment"]["department"] == "呼吸内科"

    def test_interaction_tool(self, med_server):
        out = json.loads(med_server.check_drug_interaction_tool("华法林", "阿司匹林"))
        assert out["severity"] == "高风险"

    def test_search_knowledge_tool(self, med_server):
        out = json.loads(med_server.search_knowledge_tool("医保报销"))
        assert out["success"] and out["chunks"]
