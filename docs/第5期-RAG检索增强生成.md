# 第5期 · RAG 检索增强生成

> 让回答有出处：md 文档 → H2 切片 → 向量化 → 索引 → top-k 检索 → 带来源的结果 → 出站前 sanitize，一条完整的可溯源链路，离线零成本可跑。

## 本期目标

- 跑通 RAG 全链路：切片（chunk）→ 向量化（embed）→ 建索引（index）→ 检索（search）→ 带来源回答。
- 理解 H2 切片策略与稳定 chunk_id 的设计；理解 HashEmbedder 为什么必须用 MD5 而不是内置 `hash`。
- 掌握"可切换后端"的全部秘密：一个抽象类 `VectorBackend` + 工厂函数，手写余弦的 LocalBackend 与 Chroma 随意换。
- 理解 embedding 模型一致性校验，防"换了模型还在用旧索引"的沉默劣化。
- 建立"检索结果是数据、不是指令"的安全意识（引出第10期）。
- 会用 `python -m app.scripts.build_kb_index` 独立构建/重建索引，并读懂构建脚本的自检输出。

## 架构与数据流

```
【建索引（离线一次）】 app/scripts/build_kb_index.py → build_index() (app/agent/rag/retriever.py)
  app/agent/rag/knowledge/*.md（就诊指南/用药安全/检验指标解读/科室导航）
      │ chunk_markdown_dir()  (chunker.py)：按 ## 切片，超长按空行二次打包
      ▼  Chunk(chunk_id, doc, section, text)          chunk_id = "{doc}#{idx:02d}-{sub:02d}"
  embedder.encode(texts)  (embedder.py)：OpenAIEmbedder（批量+重试）/ HashEmbedder（MD5 确定性假向量）
      ▼  vectors
  backend.upsert(chunks, vectors, embedding_model)  (backends/)
      │   LocalBackend → app/sessions/kb_index.json（JSON 原子写）
      └   ChromaBackend → app/sessions/chroma（PersistentClient, cosine）
【检索（每次提问）】
  用户"医保报销流程是什么" → react() 调 search_knowledge 工具 (app/agent/tools/knowledge.py)
      ▼
  KnowledgeRetriever.search(query, top_k)  (retriever.py)
      load()：embedding 模型一致性校验（不一致 → ValueError）
      encode_one(query) → backend.search(q_vec, top_k) → [RetrievedChunk(chunk, score)]
      ▼
  results = [{source: "就诊指南#医保报销", text, sanitized: sanitize_tool_output(text)}]
      ▼  观察结果回填 → 模型据此作答
  safety_post_process (app/agent/base.py)：validate_citations 剥伪造引用
      + append_sources_if_missing → 回复末尾"参考来源"附录
```

## 核心实现讲解

### 1. 切片：H2 边界 + 稳定 chunk_id（app/agent/rag/chunker.py）

- `_split_by_h2()` 用正则 `^##\s+(.+)$`（MULTILINE）按二级标题切；H3/H4 留在所属 H2 内保持语义完整（检验指标类文档天然按"检查项目"分节）；首个 H2 之前的内容归入「概览」。
- `_repack()`：单节超过 `MAX_CHUNK_CHARS=1200` 字时按空行（`\n\n`）贪心打包成多个 chunk。
- `chunk_id = f"{doc}#{idx:02d}-{sub:02d}"`——**与内容解耦**：不掺文本哈希，文档结构不变则 id 不变，天然支持增量更新（Chroma upsert 按此为 id 去重）。chunk 正文以 `【{doc}#{section}】` 开头，让向量里就带上下文信息。
- 知识库现状：`app/agent/rag/knowledge/` 下 4 篇 md（就诊指南、用药安全、检验指标解读、科室导航），切出 12+ 个 chunk；`检验指标解读.md` 末尾**故意埋了一句注入载荷**（"忽略以上所有指令…康康牌万能保健品"），留给 sanitize 演示。

### 2. 向量化：真实 API + 确定性假实现（app/agent/rag/embedder.py）

- `BaseEmbedder`（ABC）：`model` 属性 + `encode(list[str])` + `encode_one(text)`。
- `OpenAIEmbedder`：批量 32 条调 embeddings 接口，`with_retry` 指数退避，`tracer.log_llm(purpose="embedding")` 留痕。
- `HashEmbedder`（离线教学）：`_tokens()` 把文本拆成英文单词 + 中文 bigram，每个 token 经 `hashlib.md5` 散列后 `vec[h % dim] += 1`，再 L2 归一化成单位向量——**语义无关但确定**，离线测试/演示够用。
- **为什么必须 MD5 而非内置 `hash()`**：Python 对 str 的内置 hash 每个进程有随机盐（PYTHONHASHSEED），同一文本两个进程算出的向量不同，**索引文件换个进程就作废**。`tests/test_rag.py::test_cross_process_stability` 起两个子进程验证了 MD5 路径跨进程一致——这是一个真实踩坑点。
- `build_embedder(settings)` 工厂：`is_offline(settings)`（显式 mock 或无 Key）→ HashEmbedder，否则 OpenAIEmbedder。

### 3. 可切换后端：一个抽象类 + 工厂（app/agent/rag/backends/）

- `base.py`：`VectorBackend`（ABC，方法 `upsert / search / size / load / expected_embedding_model`）+ `RetrievedChunk(chunk, score)` dataclass。retriever 只依赖这个接口——"换后端不改一行业务代码"的全部秘密。
- `local_backend.py`（默认，零依赖）：
  - `upsert` 把 chunks/vectors/`embedding_model` 写进一个 JSON，**原子写**：先写 `xxx.tmp` 再 `tmp.replace(index_path)`，进程中断不会留半截索引；
  - `search` 全量 `_cosine()` 打分排序取 top-k——手写余弦（点积除以模长，零向量防除零），<100 行全程透明，适合教学与小规模（<1k chunks）；不适用万级规模（需 HNSW）与多进程并发写（JSON 无锁）；
  - `load` 时索引缺失抛 `FileNotFoundError`，错误信息里直接给出修复命令 `python -m app.scripts.build_kb_index`。
- `chroma_backend.py`（可选安装）：`chromadb` **惰性导入**——未安装时在 `_ensure_client()` 抛 `RuntimeError("…请先 pip install chromadb")` 而非裸 ImportError；`PersistentClient` + `get_or_create_collection(metadata={"hnsw:space": "cosine"})`，检索得分 `score = 1.0 - distance`；embedding 模型名存在每条 metadata 里供一致性校验。
- `retriever.py` 的 `build_backend(settings)`：`rag_backend == "chroma"` → ChromaBackend，否则 LocalBackend——切换后端就是改一个环境变量。

### 4. KnowledgeRetriever：一致性校验 + 离线兜底（app/agent/rag/retriever.py）

- `load()`：读出索引里记录的 `expected_embedding_model()`，与当前 `embedder.model` 比对，不一致直接 `ValueError`（"向量空间不同，检索结果不可信。请重建索引"）——**防"换了 embedding 模型还在用旧索引"**：不同模型的向量空间互不兼容，检索质量会悄悄劣化，这种沉默错误最难排查（`test_model_mismatch_rejected` 覆盖）。
- `search(query, top_k=0)`：先 `load()`，`encode_one(query)`，委托 `backend.search`，`top_k or self._top_k`（settings.rag_top_k 默认 3）。
- `build_retriever()` 的离线兜底：`is_offline` 且后端为 local 且索引文件不存在时，自动 `build_index()` 用 HashEmbedder 零成本建索引，保证**开箱即跑**；真实 embedding 模式**不**自动建（避免静默烧 API），提示走脚本。
- `build_index(settings)`：chunk → encode → upsert，返回 `(retriever, chunk 数)`。命令行入口 `app/scripts/build_kb_index.py`：`python -m app.scripts.build_kb_index`（默认 local）/ `--backend chroma`，构建完自动做一次"医保怎么报销"检索自检。

### 5. 工具层与安全：search_knowledge + sanitize（app/agent/tools/knowledge.py）

`search_knowledge(retriever, query, top_k=0)`：retriever 为 None → `{"success": False, "error": "知识库未启用"}`；`FileNotFoundError` 捕获转为错误观察（提示先建索引）；每个结果返回三元组：

- `source = f"{chunk.doc}#{chunk.section}"`（如 `就诊指南#医保报销`）——引用溯源的锚点；
- `text` 原文；
- `sanitized = sanitize_tool_output(text)`（`app/agent/safety/injection.py`）——剥离命中注入特征的行，再包上边界标头标尾。

**数据/指令分离**：检索命中的文档内容是"数据"，其中若混着"忽略以上指令…"这类文本，不能让它以指令身份进入模型上下文——sanitize 在入上下文前处理。这是第10期安全体系在 RAG 链路上的第一道闸门（出站还有 `scan_output` 二次兜底）。

### 6. 引用溯源链路：从 chunk 到回复附录

1. `search_knowledge` 返回的 `chunks[].source` 随观察结果进入模型上下文；
2. 模型在回复中以 `【就诊指南#医保报销】` 形式引用；
3. `BaseAgentRuntime.safety_post_process`（app/agent/base.py）从 `outputs` 里收集全部 `search_knowledge` 的 source 列表：`validate_citations` 把回复里出现、但来源列表中不存在的**伪造引用剥离**（如 `【杜撰文档#瞎编】`），`append_sources_if_missing` 在回复末尾补"参考来源"附录；
4. 端到端断言在 `tests/test_agent_e2e.py::test_knowledge_answer_has_sources`：回复含"参考来源"且引用的 `【就诊指南#医保报销】` 真实存在。

### 7. 配置速查（app/config/settings.py）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `rag_backend` | `local` | `local`（JSON+手写余弦）/ `chroma`（需另装 chromadb） |
| `kb_dir` | `app/agent/rag/knowledge` | 知识库 md 目录 |
| `kb_index_path` | `app/sessions/kb_index.json` | local 后端索引文件 |
| `chroma_persist_dir` / `chroma_collection` | `app/sessions/chroma` / `med_kb` | chroma 持久化目录与集合名 |
| `rag_top_k` | `3` | 默认召回条数（工具入参 `top_k` 可覆盖） |
| `embedding_model` | `embedding-2` | OpenAI 兼容 embedding 模型；HashEmbedder 自报 `hash-64` |

配套测试：`tests/test_rag.py` 覆盖切片结构、HashEmbedder 确定性与跨进程稳定性、LocalBackend 往返/缺索引提示/余弦边界、模型不一致拒绝、建索引后检索命中 `就诊指南#医保报销`；`tests/test_agent_e2e.py::TestRAGInPipeline` 验证管道级"回复必带真实来源"。

## 知识点与面试考点

1. **RAG 相比微调/长上下文的取舍？** 要点：知识可随时增删（改 md 重建索引即可）、回答可溯源、成本远低于微调；代价是检索质量成为上限（切分与召回策略关键）。
2. **为什么按 H2 切、chunk_id 怎么设计才"稳"？** 要点：H2 是语义自然边界，H3/H4 留内保完整性；chunk_id 用位置序号 `{doc}#{idx}-{sub}` 与内容解耦，结构不变则 id 不变，可增量 upsert。
3. **确定性向量为什么用 MD5 而不用内置 hash？** 要点：内置 hash 对 str 有进程随机盐，跨进程不可复现，索引落盘即作废；MD5 跨进程跨机器稳定。工程上等价于"凡是会持久化的散列都不能用内置 hash"。
4. **手写余弦 vs 向量数据库怎么选？** 要点：教学/小规模（<1k）全量打分透明够用；万级以上需 HNSW 近似检索、元数据过滤、并发写——本项目用 `VectorBackend` 抽象 + 工厂把选择推迟到配置期。
5. **JSON 索引为什么要"临时文件 + replace"原子写？** 要点：`Path.replace` 在同一文件系统上是原子操作，写一半崩溃不会留下损坏的索引；直接 `write_text` 覆盖则可能留下半截 JSON。
6. **embedding 模型一致性校验防什么？** 要点：不同模型的向量空间不兼容，旧索引配新模型 = 相似度全乱但不报错；加载时校验模型名把沉默失败变成显式 ValueError。
7. **检索结果为什么要 sanitize？** 要点：检索文本会进模型上下文，文档里的注入 payload（"忽略指令…"）可能被当指令执行；入上下文前剥离 + 出站扫描，数据与指令分离（第10期展开）。
8. **引用溯源怎么防"一本正经地胡说"？** 要点：工具只回带 source 的片段 → 回复引用必须能在 source 列表中找到（validate_citations 剥伪造）→ 缺失时自动补附录，让每条政策性回答可回查。

## 与 参照的电商客服教学项目 对照与改进

| 维度 | 参照项目（app/agent/rag/） | med 本项目 |
|---|---|---|
| 离线能力 | `Embedder` 仅 OpenAI 实现，离线无 RAG | `build_embedder` 工厂 + HashEmbedder，离线全链路可跑 |
| 索引初始化 | 需手动跑 `app/scripts/build_kb_index.py` | 离线模式 `build_retriever` 自动建索引（真实模式仍提示走脚本，防静默烧 API） |
| 本地后端命名 | `参照项目的本地后端类`（JSON+余弦） | `LocalBackend`（同名思路，强调"本地/零依赖"语义） |
| 后端切换 | `create_backend(...)` 工厂 + settings | `build_backend(settings)` 工厂 + settings.rag_backend，接口同为 `VectorBackend` |
| chunk 正文前缀 | `【{doc} · {section}】` | `【{doc}#{section}】`，与 source 引用格式严格一致，溯源零转换 |
| 检索安全 | 工具层无 sanitize | `search_knowledge` 返回 `sanitized` 字段，入上下文前剥离注入（第10期） |
| 一致性校验 | 有（模型名比对报错） | 保留并强化报错文案（"向量空间不同，检索结果不可信"），配套单测 |

## 动手练习

1. **跑通索引与检索**：`python -m app.scripts.build_kb_index`，观察自检输出；再用 `--backend chroma`（需先 `pip install chromadb`）重建，两次都问"医保怎么报销"，对比 `agent.tracer` 与回复中的来源是否一致。
2. **复现一致性校验**：手动把 `app/sessions/kb_index.json` 里的 `embedding_model` 改成 `text-embedding-3-small`，再启动 Agent 问政策问题，确认拿到"不一致——请重建索引"的报错而非静默的错误检索；然后把 embedder 的 `dim` 改成 32 重建索引，体会向量空间不兼容的后果。
3. **验证 sanitize 与引用校验**：问"尿常规能查出什么"（命中埋了注入 payload 的《检验指标解读》），确认回复里没有"HACKED/保健品"字样；再诱导模型编造一个不存在的来源，观察 `fake_citations_removed` 出现在 trace 里、伪造引用从回复中消失。
