# 第4期 · MCP 集成

> 把工具从 Agent 进程里拆出去：用 FastMCP 起一个只读的独立工具微服务（Streamable HTTP, 端口 9301），同步 Agent 通过后台线程事件循环无缝调用，连接失败自动降级本地。

## 本期目标

- 理解 MCP（Model Context Protocol）是什么、与各家 LLM 私有 function calling 的关系与差别。
- 会用 FastMCP 写一个 Streamable HTTP 的工具 server，并理解"只暴露只读工具"的权限边界设计。
- 掌握"同步代码接异步 SDK"的标准手法：后台线程跑事件循环 + `asyncio.run_coroutine_threadsafe`。
- 理解 MCP `inputSchema` 到 OpenAI `tools` 格式的转换（converter），以及桥接层的降级策略。
- 明白为什么 `requirements.txt` 把 mcp SDK 锁在 `>=1.8,<2`。

## 架构与数据流

```
┌───────────────────────── Agent 进程（同步代码） ─────────────────────────┐
│ MedicalAgent._init_components()  (app/agent/chat.py)                     │
│   build_tool_registry(...)  →  本地 ToolRegistry（9 个工具）              │
│   settings.mcp_enabled ──→ attach_mcp_tools(registry, s, tracer)         │
│                            (app/agent/tools/mcp_bridge.py)               │
│        │ 连接失败 → print 警告并 return False，纯本地照常工作（降级）      │
│        ▼ 连接成功                                                          │
│   MCPClient(settings.mcp_server_url)  (app/mcp_client/client.py)         │
│     connect() → list_tools() → mcp_tools_to_openai() (converter.py)      │
│     每个 MCP 工具包装成 ToolDef 闭包注册进同一 registry（同名覆盖本地）    │
│        │                                                                  │
│        │ react() 调用 MCP 工具时:  call_tool(name, args)                  │
│        │   asyncio.run_coroutine_threadsafe(session.call_tool(...), loop) │
│        ▼                        提交到后台线程的事件循环 ─────────────┐    │
└─────────────────────────────────────────────────────────────────────│────┘
                                                                      │
                                              HTTP (Streamable)       ▼
┌──────────────────────── 独立进程：MCP Server（mcp_server/server.py）──────┐
│ FastMCP("med-tools", host="127.0.0.1", port=9301)                        │
│ mcp.run(transport="streamable-http")  →  http://127.0.0.1:9301/mcp       │
│ 只读工具 x6：query_appointment_tool / query_medicine_tool /               │
│   check_drug_interaction_tool / query_lab_report_tool /                   │
│   query_department_tool / search_knowledge_tool                          │
│ （cancel_appointment 写操作、memory/skill 有状态工具 不出本地进程）        │
└───────────────────────────────────────────────────────────────────────────┘
```

## 核心实现讲解

### 1. MCP 是什么，解决什么问题

MCP 是一个开放协议，标准化"应用 ↔ 模型 ↔ 工具/数据源"之间的连接：客户端连上 server 后先 `list_tools()` 发现能力，再 `call_tool(name, arguments)` 调用，消息走 JSON-RPC。与 function calling 的对比：

- **function calling 是 API 约定**：OpenAI 兼容接口里的 `tools=[...]` 字段只是"模型决定调哪个函数、给什么参数"，**执行仍在客户端进程内**，且各家格式互不相同，工具换一家模型就要重写对接；
- **MCP 是进程间协议**：工具活在独立 server 里，任何支持 MCP 的客户端（本项目 Agent、其他 Agent、IDE）连上即可复用同一批工具，发现与调用都是协议标准化的。

两者不互斥：本项目的 MCP 工具被发现后，仍经 converter 转成 OpenAI function calling 格式喂给模型——模型看到的还是普通工具，底层执行被转发到远端。

### 2. FastMCP server：只读 = 权限边界（mcp_server/server.py）

- `mcp = FastMCP("med-tools", host="127.0.0.1", port=9301)`，入口 `mcp.run(transport="streamable-http")`，服务地址 `http://127.0.0.1:9301/mcp`。
- 暴露 6 个**只读**工具，命名带 `_tool` 后缀（`query_appointment_tool` 等）：函数体直接复用 `app/agent/tools/` 里的本地实现，`json.dumps(..., ensure_ascii=False)` 序列化返回。
- **为什么只暴露查询类**：微服务化意味着"任何人连上即可调用"。写操作 `cancel_appointment`（真改预约状态）与有状态工具 `recall_user_memory`/`load_skill`（依赖会话/用户上下文）留在本地进程——外发的必须是无副作用能力，这是工具的权限边界设计。
- server 走同一套 `Settings` 工厂与 `build_retriever`：无 Key 时用 HashEmbedder + 本地索引，**离线也能起服务**（启动横幅会打印 `offline=True/False`）。
- 文件顶部 `try: from mcp.server.fastmcp import FastMCP`，失败时抛出明确指引"请安装 mcp>=1.8,<2"——这是 SDK 版本锁定的第一道防线。

### 3. 同步代码接异步 SDK：后台线程事件循环（app/mcp_client/client.py）

官方 mcp SDK 是 asyncio 原生，而本项目 Agent 是同步代码。`MCPClient` 的手法：

- `connect()`：`asyncio.new_event_loop()` 新建循环，daemon 线程 `run_loop()` 里 `run_until_complete(self._run(...))`；`_run` 协程依次完成 `streamable_http_client(server_url)` 建连 → `ClientSession(read, write)` → `session.initialize()` 握手 → `session.list_tools()` 发现工具 → 结果经 `mcp_tools_to_openai()` 转换写入共享列表 → `_connected.set()` 通知主线程 → `await self._close_event.wait()` **挂起保活**（会话不随 connect 结束而关闭）。
- 主线程 `self._connected.wait(timeout=30)`；若 `_run` 里出了异常，异常对象存进 `errors` 列表带回主线程 `raise`——异步世界的错误同步地抛。
- `call_tool(name, arguments)`：`asyncio.run_coroutine_threadsafe(self._session.call_tool(name, arguments), self._loop)` 把协程**跨线程提交**给后台循环，`future.result(timeout=30)` 同步等待；`result.isError` 或超时异常都包装成 `{"error": ...}` JSON 字符串返回——和第3期的工具异常包装哲学一致。
- `close()`：`call_soon_threadsafe` 触发 `_close_event`，`join(timeout=5)` 回收线程，清空引用。

一句话总结：**循环活在线程里，会话活在循环里，调用经 run_coroutine_threadsafe 进去、结果经 future 出来**——同步代码零改动即可用 MCP。

### 4. converter：两个生态的 schema 对接（app/mcp_client/converter.py）

`mcp_tool_to_openai(tool)`：MCP 的 `inputSchema` 本身就是 JSON Schema 子集，转换只是包一层 `{"type":"function","function":{"name","description","parameters"}}` 信封；`inputSchema` 缺失时兜底 `{"type":"object","properties":{}}`（`setdefault` 两行防御）。`mcp_tools_to_openai(tools)` 是批量版。两个"工具协议"在此对接完毕。

### 5. mcp_bridge：降级优先（app/agent/tools/mcp_bridge.py）

`attach_mcp_tools(registry, settings, tracer) -> bool`：

- 连接抛异常 → 打印"⚠️ [MCP] 连接失败，降级使用本地工具"，`return False`——**MCP 是增强不是依赖**，Agent 照常用本地 9 个工具启动（`tests/test_mcp.py::TestBridgeDegradation` 用一个不可达端口 59999 验证了这一点）。
- 连接成功 → 把每个 OpenAI 格式定义包装成 `_call` 闭包（默认参数捕获 client 与工具名，避免闭包晚绑定踩坑），`registry.register(...)` 注册进**同一个 ToolRegistry**——对 react() 循环完全透明，它不知道也不需要知道工具在本地还是远端；同名工具 MCP 版覆盖本地版（服务端是事实之源）；`tracer.log("mcp_connected", tools=..., url=...)` 留痕。

### 6. 启动与联调

```bash
# 终端 1：起 MCP server（离线可跑）
python mcp_server/server.py
# → MCP server 启动: http://127.0.0.1:9301/mcp (offline=True)

# 终端 2：以 MCP 模式跑 Agent（.env 或环境变量）
#   MCP_ENABLED=1   （settings.mcp_enabled，默认 False）
#   MCP_SERVER_URL=http://127.0.0.1:9301/mcp
python main.py
# → 🔗 [MCP] 已连接 http://127.0.0.1:9301/mcp，注册 6 个工具

# 相关测试（schema 转换 / 降级 / server 工具函数）
pytest tests/test_mcp.py -v
```

注意 `tests/test_mcp.py` 里 `med_server` fixture 会先 `build_index(Settings(mock_mode=True))` 建本地索引再导入 server 模块；真正的跨进程 Streamable HTTP 联调（起 server + client）就是上面的手动实验。

### 7. 为什么锁 mcp>=1.8,<2（requirements.txt）

MCP **协议**是稳定的，但 Python **SDK 会漂移**：v2 把 `FastMCP` 改名为 `MCPServer` 等一系列破坏性变更，裸 `import mcp.server.fastmcp.FastMCP` 会直接 `ModuleNotFoundError`。所以上界锁死 `<2`，配合 server.py 顶部的友好报错，把"升级依赖后服务起不来"变成一句可执行的提示。参照项目 只写了 `mcp>=1.8.0` 无上界，是本项目修掉的隐患。

## 知识点与面试考点

1. **MCP 与 function calling 的区别？** 要点：前者是跨进程的工具发现/调用开放协议（list_tools/call_tool，JSON-RPC over Streamable HTTP），后者是单家 API 的参数格式约定、执行在客户端；本项目用 converter 把两者接起来。
2. **Streamable HTTP 是什么？** 要点：MCP 的传输层之一——单个 HTTP 端点上的双向流式会话，取代旧的 HTTP+SSE 双端点；server 只需暴露一个 `/mcp` URL。
3. **同步代码怎么调 asyncio SDK？** 要点：专用后台线程 `run_until_complete` 跑循环并保活，调用经 `run_coroutine_threadsafe` 提交、`future.result` 等待；关闭经 `call_soon_threadsafe` 置事件。注意不能在主线程已经有循环时嵌套 `run_until_complete`。
4. **为什么 MCP server 只暴露只读工具？** 要点：网络可达 = 任何人可调；写操作与用户态工具出进程等于交出权限。权限边界按"副作用"划线，而不是按"重要程度"。
5. **桥接失败为什么要降级而不是报错退出？** 要点：MCP 是能力增强不是硬依赖；降级路径要被测试覆盖（不可达端口用例），否则"平时全靠 MCP、一挂全站瘫"。
6. **同名工具冲突怎么处理？** 要点：MCP 版覆盖本地版——服务端是事实之源，避免本地旧实现与远端行为漂移。
7. **依赖为什么要锁版本区间？** 要点：协议稳定 ≠ SDK 稳定；`>=1.8,<2` + 导入期 try/except 报错指引，把破坏性升级挡在门外。

## 与 参照的电商客服教学项目 对照与改进

| 维度 | 参照项目（app/mcp_client/、mcp_server/server.py） | med 本项目 |
|---|---|---|
| 工具聚合 | `ToolManager` 类聚合本地+MCP 两套来源，`execute_tool` 按 `_tool_source` 分发 | MCP 工具直接注册进 `ToolRegistry`，对 react() 透明，无第二套分发逻辑 |
| 白名单 | `ToolManager._filter_tools`（与注册表白名单分两处） | 统一走 `ToolRegistry.subset()` |
| server 暴露面 | 查询类 + **`apply_refund` 写操作**（端口 9123） | 仅 6 个只读工具（端口 9301），写/有状态工具不出进程 |
| SDK 版本约束 | `mcp>=1.8.0`（无上界，v2 升级即断） | `mcp>=1.8,<2` + 导入期友好报错 |
| 客户端实现 | 同款后台线程手法（med 沿用并微调） | `result.isError` 改用 `getattr` 防御属性缺失，错误文案区分"调用失败/执行出错" |
| 联调体验 | 依赖外部说明 | server 横幅打印 offline 状态；`tests/test_mcp.py` 明确划分离线单测与手动联调 |

## 动手练习

1. **完整联调一轮**：按"启动与联调"起 server，`MCP_ENABLED=1` 跑 `python main.py`，问"帮我查一下预约 GH-2026-001"；然后 Ctrl+C 杀掉 server 再问一次，确认 Agent 打印降级警告且回复照常（本地工具兜底）。
2. **观察覆盖行为**：起 server 后问"布洛芬怎么吃"，对比 `app/sessions/trace.jsonl` 里 `mcp_connected` 与工具调用记录，说明这条调用走的是远端 `query_medicine_tool`；再注释掉 `attach_mcp_tools` 里的同名覆盖逻辑，思考本地版与远端版行为可能如何漂移。
3. **加一个只读 MCP 工具**：在 server.py 仿照现有工具加 `list_departments_tool()`（返回 `DEPARTMENTS` 全表），重启 server 与 Agent，验证新工具被自动发现并注册——体会"server 加工具，客户端零改动"的协议价值。
