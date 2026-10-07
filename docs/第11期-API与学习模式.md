# 第11期 API服务与学习模式前端

把 CLI 版「小医」搬到浏览器：`app/api.py`（FastAPI + SSE）+ `static/index.html`（学习模式前端）。
与 legal-service-agent 同构，本文只记启动方法与医疗版差异。

## 启动

```bash
# conda activate med-agent
uvicorn app.api:app --host 127.0.0.1 --port 8305
# 浏览器打开 http://127.0.0.1:8305/ （默认 mock 离线，零 API 成本）
# 真实 LLM：服务端先 export ALLOW_REAL_MODE=1，前端再选 real 模式（服务端锁定，见下）
```

端点：`GET /health` · `POST /chat` · `GET /chat/stream`(SSE) · `POST /reset` · `/static`、`/docs` 挂载。
会话管理：前端生成 uuid 作 session_id，每会话独立文件（`app/sessions/api/{sid}.json`）+
`memory_user_id=session-{sid}` 记忆命名空间隔离；同会话 per-session 锁串行；reset 走
`reset_session()` 真语义（清历史/摘要/STM/会话文件，长期记忆保留）。

## Docker 启动

与上节二选一。仓库根有 `Dockerfile` / `.dockerignore` / `docker-compose.yml`
三件套（⚠️ 未在本机构建实测——编写时本机无 Docker）：

```bash
docker compose up -d --build        # 构建并后台启动（无 .env 时自动进离线 mock）
curl http://127.0.0.1:8305/health   # 期望 {"status":"ok","offline":true,...}
docker compose down                 # 停止并移除容器
```

要点：镜像 `python:3.11-slim` + `pip install -r requirements.txt`，`COPY .`
时 `.dockerignore` 排除 `.env`/`**/.env`（密钥绝不进镜像层）、`.git`、
`__pycache__`、`.pytest_cache` 与 `app/sessions`（运行时产物）；密钥由
compose 的 `env_file`（可选，`.env` 缺失不报错）运行时注入，真实模式还需
在 `.env` 中加 `ALLOW_REAL_MODE=1`。会话/记忆/索引落在具名卷
`med-sessions`，`restart: unless-stopped` 保证崩溃/重启自愈。

## 与 legal 版的差异点

| 项 | legal | med |
|---|---|---|
| Agent | `LegalAgent` | `MedicalAgent`（`app.agent.chat`），响应徽章为 intent/urgency/requires_human |
| 端口/CORS | 8300 | 8305 |
| SSE 事件白名单 | 含 `nli_check`/`llm_fallback` | `llm_call/tool_call/route/redflag_shortcut/injection_blocked/fake_citations_removed`（med 无 NLI 与降级环节，grep tracer 确认） |
| 管线①语义 | 人身安全红旗→110 | 急症红线（胸痛/呼吸困难→120），命中零 LLM 短路 |
| 管线④语义 | 注入/引用/NLI | 注入扫描/引用校验/危急值升级转人工 |

前端学习模式四件套不变：实时管线条（7节点亮灯+被短路标签）、事件教学注解、
引导实验（观察点+完成后自动核对）、指标小结+复盘重播；全部数据 `textContent` 渲染。

## 六个引导实验

| chip | 消息 | 观察点 | 自动核对 |
|---|---|---|---|
| 查预约(两段式ReAct) | 帮我查一下预约 GH-2026-001 | ③亮🔧1；llm→tool(query_appointment)→llm；回复含「呼吸内科」 | tool名/2次llm/呼吸内科 |
| 急症红线(零LLM) | 我突然胸痛 | 只有①亮③④被短路；零llm_call；回复含120 | 零llm/redflag事件/含120/requires_human |
| 用药咨询 | 布洛芬怎么吃 | tool=query_medicine；回复含布洛芬用法 | 两项核对 |
| 危急值报告 | 帮我看看报告 LAB-2026-004 | tool=query_lab_report；回复含「危急值」；转人工=是 | 三项核对 |
| 医保政策(RAG) | 医保报销流程是什么 | tool=search_knowledge；回复带【就诊指南…】或「参考来源」 | 两项核对 |
| 链式写操作 | 取消我的预约 GH-2026-001 | 两个tool: query_appointment→cancel_appointment | 链式顺序/≥2次工具 |

核对文案以 med 的 MockLLM 实测行为为准（mock 的 `parse_structured` 不记 llm_call，
故两段式 ReAct 的 llm_call 计数为 2）。测试见 `tests/test_api.py`。
