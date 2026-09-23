# med-service-agent：医疗健康导诊 Agent（从0到1教学实现）

> 以「互联网医院健康助手」为场景，从0构建的**企业级 Agent 技术栈教学项目**：
> Prompt工程 → 结构化输出 → 多轮对话 → ReAct → 工具调用 → MCP → RAG →
> Multi-Agent → Memory → Skill → 评估体系 → 安全护栏。
> **全部模拟数据、离线可跑**（零 API 成本），配好 Key 即接真实模型（GLM/DeepSeek/OpenAI）。

## 快速开始（conda 环境）

```bash
# 1) 创建环境并安装依赖（Python 3.11）
conda create -n med-agent python=3.11 -y
conda activate med-agent
pip install -r requirements.txt        # mcp 锁 v1（v2 有破坏性 API 变更）

# 2) 离线跑起来（不需要任何 API Key）
python main.py --mock                  # 单 Agent 模式
python main.py --mock --multi          # Multi-Agent 模式（挂号/用药/报告/急诊分流）

# 3) 跑测试（135 个用例，全离线）
pytest

# 4) 离线评估（17 条黄金用例，含 3 条急症安全红线）
python -m app.scripts.build_kb_index   # 构建知识库索引（离线自动建，此命令显式重建）
python -m app.scripts.run_eval         # 规则模式评估（零成本）
python -m app.scripts.run_eval --mode multi

# 5) MCP 联调（可选）
python mcp_server/server.py            # 另起终端，FastMCP @ 127.0.0.1:9301/mcp
# .env 中 MCP_ENABLED=true 后重启 main.py，工具改经 MCP 协议调用，失败自动降级本地
```

试试这些输入：`查预约 GH-2026-001`、`华法林和阿司匹林能一起吃吗`、
`帮我看看报告 LAB-2026-004`（危急值演示）、`医保报销流程`、`我突然胸痛`（急症红线短路）。
会话命令：`skills` 查看技能 · `memory` 查看记忆 · `summary` 查看压缩摘要 · `reset` 重置。

## 技术栈与项目结构

| 期 | 技术 | 核心代码 |
|---|---|---|
| 1 | 结构化输出 + 重试退避 LLM 客户端 | `app/schemas/response.py`、`app/llm/client.py` |
| 2 | 多轮对话：摘要压缩 + 原子持久化 | `app/agent/storage.py`、`summarizer.py` |
| 3 | ReAct + 9 个医疗工具（含写操作/危急值） | `app/agent/base.py`、`app/agent/tools/` |
| 4 | MCP：FastMCP server + 同步 client + 降级 | `mcp_server/server.py`、`app/mcp_client/` |
| 5 | RAG：H2切片 + 双后端(手写余弦/Chroma) + 一致性校验 | `app/agent/rag/`、`knowledge/*.md` |
| 6 | Multi-Agent：路由 + 四子Agent工具白名单 | `app/multi_agent/` |
| 7 | Memory：STM/LTM + structured 提取 + 防投毒 | `app/agent/memory/` |
| 8 | Skill：SKILL.md + 渐进式披露 | `app/agent/skills/` |
| 9 | 评估：过程/结果×规则/judge 2×2 + 沙箱 | `app/evaluation/` |
| 10 | 安全护栏：红旗短路/注入防御/引用校验 | `app/agent/safety/` |

```
med-service-agent/
├── main.py                    # CLI 入口（--mock / --multi）
├── app/
│   ├── config/settings.py     # 可多实例配置（沙箱不污染全局）
│   ├── llm/client.py          # LLM 抽象：重试退避 + Mock（离线可跑的关键）
│   ├── agent/
│   │   ├── base.py            # ★ 公共运行时：ReAct/会话/提取/安全后处理
│   │   ├── chat.py            # 单 Agent：小医
│   │   ├── tools/             # 9 工具 + 注册表(白名单/异常包装) + MCP桥
│   │   ├── rag/               # chunker/embedder(真+假)/双后端/retriever
│   │   ├── memory/            # STM/LTM/提取(structured)/防投毒校验
│   │   ├── skills/            # 3 个 SKILL.md + 渐进式披露 loader
│   │   ├── safety/            # 红旗/注入/引用 三道护栏
│   │   └── tracer.py          # 全链路 token/工具/路由 计量
│   ├── multi_agent/           # Router + 四子Agent + 编排器
│   ├── evaluation/            # 沙箱 + 2×2 指标 + 17条用例(含安全红线)
│   └── scripts/               # build_kb_index / run_eval
├── mcp_server/server.py       # FastMCP（Streamable HTTP, 只读工具, :9301）
├── tests/                     # 135 个离线测试
└── docs/                      # 第0~10期教学文档（学习主路径）
```

## 学习路径

1. **跑起来**：`python main.py --mock`，对照每轮输出里的 `[意图|置信度|紧急度|转人工]` 与事件轨迹；
2. **按期读文档+代码**：`docs/第0期` → `第10期`，每篇含架构图、设计取舍、面试考点、与 参照的电商客服教学项目 的对照改进；
3. **改代码做练习**：每期文档末尾有动手练习（建议至少做 3 个）；
4. **接真实模型**：`.env` 填 `OPENAI_API_KEY`（默认智谱 GLM，可换任意 OpenAI 兼容接口），去掉 `--mock` 即真实模式；
5. **评估守护**：改完 prompt/模型后跑 `python -m app.scripts.run_eval`，防退化。

## 与参照项目 参照的电商客服教学项目 的关键差异

| 方面 | 参照项目 | 本项目 |
|---|---|---|
| 测试 | 33 个用例直连真实 API，无 Key 全挂 | 135 个全离线（MockLLM+FakeEmbedder），零成本 |
| 全局状态 | 模块级 setter 单例，沙箱改全局不还原 | 全部构造注入，沙箱独立 Settings 实例 |
| LLM 容错 | 无重试 | 统一重试退避（仅瞬态错误） |
| 可观测 | 仅评估时 monkey-patch 采集 | tracer 一等公民，运行时即计量 |
| 代码复用 | 单Agent/编排器 ~200 行复制粘贴 | 公共 BaseAgentRuntime，ReAct 全系统唯一实现 |
| 安全 | Guardrails 列为"待做" | 红旗短路/注入防御/引用校验/记忆防投毒已实现+测试 |

## 诚实声明（面试主动交代反而加分）

- 全部为**模拟数据**（预约/药品/报告/知识库均为教学虚构），非真实医疗数据，**不做诊断**只做导诊咨询；
- 离线 mock 模式的内容是脚本化应答（但会真实走工具/记忆/安全链路，轨迹结构与真实模式一致）；
- 检索用确定性假向量（离线）或真实 embedding（配 Key），无 rerank/混合检索（升级练习见 docs 第5期）；
- 红旗规则词表存在已知误报（宁可误报不可漏报的取舍，见 docs 第10期"残余风险"）。
