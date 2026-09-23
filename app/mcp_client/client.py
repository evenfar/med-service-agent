"""MCP Client（第4期）：Streamable HTTP + 后台线程同步封装。

难点：官方 mcp SDK 是 asyncio 原生，而本项目 Agent 是同步代码。
方案：专用后台线程跑事件循环，connect 后保持会话存活，工具调用经
run_coroutine_threadsafe 提交到该循环 —— 同步代码零改动即可用 MCP 工具。
（这是"同步代码接异步生态"的标准手法，面试可讲线程/事件循环边界。）
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional

from app.mcp_client.converter import mcp_tools_to_openai


class MCPClient:
    def __init__(self, server_url: str):
        self._server_url = server_url
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._connected = threading.Event()
        self._close_event: Optional[asyncio.Event] = None

    def connect(self) -> list[dict]:
        """连接 + 发现工具。返回 OpenAI 格式工具定义列表。"""
        self._loop = asyncio.new_event_loop()
        self._close_event = asyncio.Event()
        tool_definitions: list[dict] = []
        errors: list[Exception] = []

        def run_loop():
            self._loop.run_until_complete(
                self._run(tool_definitions, errors))

        self._thread = threading.Thread(target=run_loop, daemon=True,
                                        name="mcp-client-loop")
        self._thread.start()
        self._connected.wait(timeout=30)
        if errors:
            raise errors[0]
        return tool_definitions

    async def _run(self, tool_definitions: list[dict], errors: list[Exception]):
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client(self._server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    result = await session.list_tools()
                    tool_definitions.extend(mcp_tools_to_openai(result.tools))
                    self._connected.set()
                    await self._close_event.wait()
        except Exception as e:  # noqa: BLE001 —— 异常带回调用线程
            errors.append(e)
            self._connected.set()

    def call_tool(self, name: str, arguments: dict) -> str:
        if not self._session or not self._loop:
            return json.dumps({"error": "MCP 客户端未连接"}, ensure_ascii=False)
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop)
        try:
            result = future.result(timeout=30)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"MCP 调用失败: {e}"}, ensure_ascii=False)
        if getattr(result, "isError", False):
            text = result.content[0].text if result.content else "未知错误"
            return json.dumps({"error": f"MCP 工具执行出错: {text}"},
                              ensure_ascii=False)
        return result.content[0].text if result.content else "{}"

    def close(self) -> None:
        if self._close_event and self._loop:
            self._loop.call_soon_threadsafe(self._close_event.set)
        if self._thread:
            self._thread.join(timeout=5)
        self._session = None
        self._loop = None
        self._thread = None
