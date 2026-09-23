"""MCP 工具 schema → OpenAI function calling 格式转换（第4期）。

MCP 的 inputSchema 本身就是 JSON Schema 子集，转换主要是包一层
{"type":"function","function":{...}} 信封 —— 两个生态的"工具协议"在此对接。
"""

from __future__ import annotations


def mcp_tool_to_openai(tool) -> dict:
    name = getattr(tool, "name", "") or ""
    description = getattr(tool, "description", "") or ""
    schema = dict(getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": schema}}


def mcp_tools_to_openai(tools) -> list[dict]:
    return [mcp_tool_to_openai(t) for t in tools]
