# 第6期 · Multi-Agent 协作：路由器、子 Agent 白名单与编排器复用

> 用"Router + 四个子 Agent"拆分单一职责，并让单 Agent 与多 Agent 共享同一份运行时骨架 BaseAgentRuntime，消除 参照项目里约 200 行的复制粘贴。

## 本期目标

- 理解意图路由器的工程细节：temperature=0 + max_tokens=10 + 受控词表匹配兜底；
- 掌握"红旗短路先于路由"的优先级设计：急症走规则，不走 LLM；
- 学会用工具白名单（registry.subset）+ prompt 声明把"权限最小化"落到两层；
- 看懂编排器如何继承 BaseAgentRuntime，把会话/压缩/提取/安全全部复用；
- 能回答"什么时候该拆多 Agent，什么时候单 Agent 更好"。

## 架构与数据流

```text
用户输入
   │
   ▼
┌─────────────────────────────┐
│ BaseAgentRuntime.handle_redflag   app/agent/base.py
│ 规则命中急症红线？ ──是──▶ 直接返回急诊应答（零 LLM 调用，短路退出）
└──────────────┬──────────────┘
               否
               ▼
┌─────────────────────────────┐
│ Router.route                app/multi_agent/router.py
│ LLM 分类（temperature=0,    │
│ max_tokens=10）→ 词表匹配    │
│ → 兜底 DEFAULT_AGENT        │
└──────────────┬──────────────┘
               ▼ 四选一：SUBAGENT_SPECS   app/multi_agent/agents.py
   ┌───────────┬───────────┬───────────┐
   ▼           ▼           ▼           ▼
appointment medication  report     emergency
挂号分诊      用药咨询    报告解读    急诊通道（tools=frozenset()，零工具）
   │           │           │           │
   └─────► _full_registry.subset(白名单)   app/agent/tools/registry.py
               │
               ▼
┌─────────────────────────────┐
│ BaseAgentRuntime.react      app/agent/base.py（全系统唯一 ReAct 实现）
│ system = 子Agent prompt + 技能目录；purpose="react:xxx" 打标签
└──────────────┬──────────────┘
               ▼
safety_post_process（注入扫描/引用校验/危急值置顶）
               ▼
extract_structured → MedicalResponse → finish_turn（记忆/压缩/持久化）
```

## 核心实现讲解

### app/multi_agent/router.py —— 意图路由器

`Router.route(user_input, history)` 三个关键决策：

1. **temperature=0.0 + max_tokens=10**：分类任务要的是确定性，不需要采样自由度；
   输出只有一个词，给 10 个 token 足够，还能防模型"解释起来没完"。
2. **受控词表包含匹配 + 默认兜底**：拿到回复后 `raw.strip().lower()`，遍历
   `VALID_AGENTS = ("appointment", "medication", "report", "emergency")` 做
   `if agent in raw` 包含匹配；全不命中返回 `DEFAULT_AGENT = "appointment"`。
   模型多吐一个标点、多写一句话都不会崩——解析永远有出口。
3. **system/user 消息分离**：`ROUTER_PROMPT` 与最近对话拼进 system，
   当前消息单独放 user。指令与数据互不污染（也是 mock 可测的前提）。
   最近对话只取 user/assistant 角色、内容 <200 字的末 4 条，仅用于消解
   指代（如"那个报告"），最终还是按当前消息分类。

### 红旗短路为什么在路由之前（优先级）

`MultiAgentOrchestrator.chat()` 第一行就是 `handle_redflag(user_input)`
（实现在 `app/agent/base.py`）。规则表 `detect_red_flags` 命中急症红线时
直接构造 emergency 应答返回：**零延迟、零漏报、不消耗一次 LLM 调用**。
此时连 Router 都不会执行——`tests/test_multi_agent.py::test_redflag_before_router`
断言 `tracer.llm_calls == []` 且没有任何 route 记录。

那 Router 的 emergency 分类还需要吗？需要：规则表覆盖典型表述，
但用户表达千变万化，LLM 分类是对规则表的兜底，两层防御。

### app/multi_agent/agents.py —— 子 Agent 配置与工具白名单

`SUBAGENT_SPECS: dict[str, SubAgentSpec]`（frozen dataclass，字段
key/name/prompt/tools）集中声明四个子 Agent。权限最小化落到两层：

| 子 Agent | 职责 | 工具白名单 |
|---|---|---|
| appointment 挂号分诊助理 | 查/退/改预约、科室推荐 | query_appointment, cancel_appointment, query_department, search_knowledge, recall_user_memory, load_skill |
| medication 用药咨询助理 | 药品说明、联用风险 | query_medicine, check_drug_interaction, recall_user_memory, search_knowledge |
| report 报告解读助理 | 检验报告解读 | query_lab_report, query_department, search_knowledge, load_skill |
| emergency 急诊通道助理 | 急症快速指引 | **无（frozenset()，零工具）** |

- 第一层（硬约束）：`ToolRegistry.subset(白名单)` 生成只含被允许工具的
  视图（共享同一批 ToolDef，不复制实现）。用药助理根本"看不见"
  cancel_appointment，prompt 说服它也没用——工具不在注册表里。
- 第二层（软约束）：`app/prompts/agents.py` 里每个子 Agent 的五段式
  prompt（角色定位/能力范围/可用工具/回复规范/安全边界）把同样的白名单
  再声明一遍，让模型不浪费步数去"尝试调用不存在的工具"。
- emergency 零工具是刻意设计：急症场景不允许把时间花在工具调用上，
  强制走固定结构的三段式快速回复。

### app/agent/base.py —— 编排器复用的基石（本篇核心）

`BaseAgentRuntime` 承载"一次对话回合的骨架"：会话持久化（save/
reset_session/load_session 恢复）、消息组装（render_messages：system →
记忆段落 → 摘要 → 历史）、**全系统唯一一份 ReAct 循环**（react）、
结构化提取（extract_structured）、安全（handle_redflag /
safety_post_process）、历史压缩（compress_history / maybe_compress）、
回合收尾（finish_turn：追加 assistant 消息 → 更新短期记忆 → 压缩 → 保存）。

留给子类的只有两个钩子：`_init_components()`（装配 memory/skills/tools，
在会话恢复之前调用）与 `_restore_extra()`（恢复短期记忆等额外状态）。

### app/multi_agent/orchestrator.py —— 编排器只剩"中间段"

`MultiAgentOrchestrator(BaseAgentRuntime)` 全文 78 行：

- `_init_components`：建 Router、MemoryManager、SkillManager、检索器，
  **全量注册一次**工具（`build_tool_registry(ToolDeps(...))`），各子 Agent
  用时再 `subset` 取视图——不是每个子 Agent 建一份注册表；
- `chat()`：红旗短路 → 路由 → 取 spec → 拼 system（spec.prompt +
  技能目录）→ `self.react(system, subset_registry, purpose=f"react:{agent_key}")`
  → `safety_post_process` → `extract_structured` → `finish_turn`。
  purpose 打上子 Agent 标签（如 `react:medication`），tracer 里可按 Agent
  分析调用轨迹，mock 客户端也靠它区分场景。

单 Agent（`app/agent/chat.py` 的 `MedicalAgent`）同样继承基类，`chat()`
与编排器逐行同构，只是中间段换成"直接用全量工具跑 ReAct"。
这就是模板方法模式：骨架一份，中间段各自实现。

### 单 Agent vs 多 Agent 的取舍

- 拆分的收益：每个 prompt 短且专注（遵循"指令越短越不容易走样"）；
  工具白名单把误操作面收窄；职责边界清晰便于单独评测。
- 拆分的成本：路由多一次 LLM 调用（延迟/费用）；路由错误会整体走错
  方向（所以兜底与最近对话消解很重要）；跨 Agent 的多轮上下文要靠
  共享 raw_messages 维护。
- 经验法则：领域可清晰划分、各域工具集差异大、安全等级不同（如急诊）
  时拆；问题域高度重叠、对话以闲聊为主时，单 Agent + 技能模块更划算。

## 知识点与面试考点

1. **路由器为什么要 temperature=0 和 max_tokens=10？**
   分类要确定性输出；max_tokens 限小既省钱又防止模型输出解释性长文，
   干扰下游解析。
2. **词表包含匹配 + 默认兜底解决了什么问题？**
   LLM 输出天然不稳定（多标点/多句话/大小写），受控词表匹配保证
   "输出再多也能解析出合法类别"，兜底保证"全不命中也不抛异常"。
3. **红旗短路为什么放在路由之前而不是之后？**
   安全检查优先级最高：规则判定零延迟零漏报，不消耗 LLM 调用；
   若先路由，急症至少多付一次分类延迟。
4. **工具白名单为什么要"注册表 + prompt"双层？**
   注册表 subset 是硬约束（模型调不到白名单外工具），prompt 声明是
   软约束（模型不浪费步数尝试）；只做软约束可被 prompt 注入绕过。
5. **emergency 子 Agent 为什么零工具？**
   急症响应的瓶颈是速度与确定性；固定结构回复优于任何工具查询，
   "无工具"在协议层面强制了快速路径。
6. **编排器继承 BaseAgentRuntime 消除了什么风险？**
   同一段会话/压缩/提取代码维护多份（参照项目 是 2~3 份），改一处漏一处；
   骨架收敛到一份后，子 Agent 只写差异部分（模板方法模式）。
7. **多轮对话里 Router 怎么处理"那个报告"这类指代？**
   取最近 4 条短消息附在 system 供消解指代，但明确要求按当前消息分类，
   防止历史喧宾夺主。
8. **如何验证子 Agent 没越权用工具？**
   测试 `test_subagent_only_uses_whitelisted_tools`：跑一轮对话后断言
   tracer 记录的全部工具调用 ⊆ 该子 Agent 白名单。

## 与 参照的电商客服教学项目 对照与改进

| 维度 | 参照项目（D:\Files\github\参照的电商客服教学项目） | med（本项目） |
|---|---|---|
| 会话骨架 | 参照项目的单Agent类（app/agent/chat.py，273 行）与 MultiAgentOrchestrator（app/multi_agent/orchestrator.py，227 行）各复制一份：提取降级、压缩、_build_messages、save/reset 几乎逐行重复，约 200 行 ×2 | 抽出 BaseAgentRuntime（app/agent/base.py）一份；MedicalAgent 与 MultiAgentOrchestrator 只写 _init_components 钩子与 chat 中间段（编排器全文 78 行） |
| ReAct 实现 | 2 份：参照项目的单Agent类._react_loop 与 SubAgent.handle（app/multi_agent/agents.py，内嵌 ~65 行循环） | 1 份：BaseAgentRuntime.react，子 Agent 复用（靠 purpose 参数区分场景） |
| 提取 fallback | _extract_structured_fallback 手写"输出 JSON"prompt + markdown 剥离，两个类里重复 | 客户端 parse_structured 统一兜底，基类只调一次 |
| 路由消息组装 | recent_context 与 ROUTER_PROMPT.format(user_input) 全拼进单条 user 消息 | system（规则+最近对话）/ user（当前消息）分离 |
| 红旗短路 | 无 | handle_redflag 在路由之前，规则命中零 LLM 调用 |
| 安全后处理 | 无 | safety_post_process（注入扫描/引用校验/危急值置顶）复用于单/多 Agent |
| 工具隔离 | 每个 SubAgent 各建一个 ToolManager 实例 | 全量注册一次，registry.subset 出白名单视图（共享 ToolDef） |
| 特殊子 Agent | 三个客服子 Agent 均配工具 | emergency 明确零工具，强制模板式快速回复 |

## 动手练习

1. **加第五个子 Agent（体检套餐咨询 checkup）**：在
   `app/prompts/agents.py` 写五段式 prompt，在 SUBAGENT_SPECS 注册并
   圈定白名单（建议只给 search_knowledge + query_department），扩展
   Router 的 VALID_AGENTS，最后在 `tests/test_multi_agent.py` 补路由
   与白名单用例。
2. **验证兜底路径**：给 Router 换一个"总输出解释性长句"的假 client，
   断言 route() 仍返回 DEFAULT_AGENT 而不抛异常；再构造一条不含任何
   关键词的闲聊（如"今天天气不错"）跑通 test_fallback_default 的逻辑。
3. **量化复用收益**：用 `wc -l` 对比 参照项目的 chat.py + orchestrator.py +
   agents.py 与本项目的 base.py + chat.py + orchestrator.py，写一段
   100 字以内的结论：哪些行消失了，为什么维护成本降低。
