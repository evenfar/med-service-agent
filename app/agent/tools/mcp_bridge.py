"""MCP 桥接（第4期）：把 MCP Server 的工具注册进本地 ToolRegistry。

对比 ecom 的改进：ecom 用独立的 ToolManager 类聚合两套来源；
本项目统一进 ToolRegistry —— MCP 工具被包装成普通 ToolDef（call_fn 闭包
转发到 MCPClient），对 ReAct 循环完全透明。连接失败自动降级纯本地工具。
"""

from __future__ import annotations

from app.agent.tools.registry import ToolRegistry
from app.agent.tracer import Tracer
from app.config.settings import Settings


def attach_mcp_tools(registry: ToolRegistry, settings: Settings,
                     tracer: Tracer | None = None) -> bool:
    """尝试连接 MCP server 并注册其工具。返回是否成功。"""
    try:
        from app.mcp_client.client import MCPClient
        client = MCPClient(settings.mcp_server_url)
        defs = client.connect()
    except Exception as e:  # noqa: BLE001 —— 连接失败降级，不阻断启动
        print(f"⚠️  [MCP] 连接失败（{e}），降级使用本地工具")
        return False

    count = 0
    for d in defs:
        fn = d["function"]
        client_ref = client

        def _call(_c=client_ref, _n=fn["name"], **kwargs):
            return _c.call_tool(_n, kwargs)

        # 同名工具：MCP 版本覆盖本地版本（服务端是事实之源）。
        # 工具名必须与本地一致——安全后处理按名匹配（危急值/引用校验）。
        registry.register(fn["name"],
                          fn["description"] or f"(MCP) {fn['name']}",
                          fn.get("parameters") or {"type": "object", "properties": {}},
                          _call)
        count += 1
    registry.mcp_client = client  # 供 agent.close() 统一释放（防线程/会话泄漏）
    if tracer:
        tracer.log("mcp_connected", tools=count, url=settings.mcp_server_url)
    print(f"🔗 [MCP] 已连接 {settings.mcp_server_url}，注册 {count} 个工具")
    return True


def close_mcp(registry: ToolRegistry) -> None:
    """关闭挂在该注册表上的 MCP 连接（若存在）。"""
    client = getattr(registry, "mcp_client", None)
    if client is not None:
        client.close()
        registry.mcp_client = None
