# 第3期 · ReAct 与工具调用

> 让 Agent 学会"先查再说"：一个唯一的 ReAct 循环 + 9 个带医疗安全设计的工具，思考与行动交替，直到给出有依据的最终回复。

## 本期目标

- 理解 ReAct（Reason + Act）循环：模型思考 → 发起工具调用 → 工具结果作为"观察"回填上下文 → 继续思考 → 直到不再调用工具、输出最终答案。
- 掌握 OpenAI function calling 的工具协议：`(name, description, JSON Schema parameters, fn)` 四元组如何变成 `tools=[...]` 参数。
- 学会把"工具崩溃"降级为"观察结果"，让模型有机会自我纠正，而不是整个 Agent 崩掉。
- 理解医疗场景特有的工具设计：写操作真改状态、"未收录"不等于"无风险"、危急值显式标记、导诊失败不瞎猜。
- 掌握 mock 数据的"状态成对覆盖"设计：让每条代码分支都有可触发的测试用例。

## 架构与数据流

```
用户输入 "帮我查一下预约 GH-2026-001"
        │
        ▼
MedicalAgent.chat()  (app/agent/chat.py)
        │  ① 红旗短路（第10期规则，命中急症则不进 LLM）
        ▼
BaseAgentRuntime.react()  (app/agent/base.py)  ← 全系统唯一 ReAct 循环
        │
        │   ┌─────────────────── 循环体（最多 max_react_steps=6 轮）─────────────────┐
        │   │ client.chat(messages, tools=registry.definitions)                      │
        │   │   ├─ 无 tool_calls → 返回最终文本，循环结束 ✔                          │
        │   │   └─ 有 tool_calls → 依次: registry.execute(name, arguments)           │
        │   │        （app/agent/tools/registry.py：校验→执行→异常包装为 JSON）        │
        │   │        观察结果以 {"role":"tool", "content": out} 回填 raw_messages     │
        │   └──────────────────────────────────────────────────────────────────────┘
        ▼
safety_post_process → extract_structured → finish_turn（chat.py / base.py）
        │
        ▼
MedicalResponse（reply / intent / urgency / requires_human ...）
```

工具实现分布在 `app/agent/tools/` 下：`registry.py`（协议与执行入口）、`mock_data.py`（模拟数据）、`appointment.py`、`medicine.py`、`lab.py`、`department.py`、`knowledge.py`。测试见 `tests/test_tools.py` 与 `tests/test_agent_e2e.py`。

## 核心实现讲解

### 1. react()：全系统唯一的 ReAct 循环（app/agent/base.py）

`BaseAgentRuntime.react(system_content, registry, purpose="react", max_steps=None)` 返回三元组 `(最终文本, used, outputs)`：

- `used: list[str]`——本轮调用过的工具名（供安全后处理判断"是否用过 search_knowledge"）；
- `outputs: list[tuple[str, str]]`——`[(工具名, 原始输出JSON)]`（供引用校验、危急值升级等下游使用）。

每轮先 `client.chat(self.render_messages(system_content), tools=tools)`：

- 模型不发起工具调用 → 把 assistant 消息入历史并返回，这是正常出口；
- 模型发起调用 → 把带 `tool_calls` 的 assistant 消息入历史（参数经 `_dump_args` 序列化），再对每个调用执行 `registry.execute(name, arguments)`，结果作为 `{"role": "tool", "tool_call_id": tc["id"], "content": out}` 回填——这就是"观察"。

**步数上限与强制收尾**：`steps = max_steps or self.settings.max_react_steps`（默认 6）。若循环耗尽模型还在要工具，代码不会粗暴截断，而是**去掉 tools 再给一次收尾机会**——`client.chat(...)` 不传 `tools`，模型只能基于已有观察作答，防止"截断在半路"输出半成品。

设计动机写在 base.py 模块注释里：参照项目 把同样的循环复制粘贴在 `参照项目的单Agent类` 与 `MultiAgentOrchestrator` 两处，改一处漏一处；本项目把"一个对话回合的骨架"抽进基类，单 Agent、编排器、SubAgent 全部复用这一份实现。

### 2. ToolDef / ToolRegistry（app/agent/tools/registry.py）

- `ToolDef` 是 frozen dataclass：`name / description / parameters / fn`，`openai_schema` 属性直接产出 `{"type":"function","function":{...}}`。**schema 与实现绑在同一条记录上，单一事实来源**——参照项目 是模块级 `TOOL_DEFINITIONS` 列表与 `_TOOL_MAP` 字典分开维护，加工具要改两处。
- `execute()` 是唯一执行入口，**永不抛异常**，流程：
  1. 未知工具 → `{"error": "未知工具: ...，可用: [...]"}`；
  2. 必填参数缺失 → `{"error": "缺少必填参数: ..."}`；
  3. 类型校验（`_TYPE_CHECK` 表，注意 `integer` 排除 `bool`，因为 Python 里 `bool` 是 `int` 子类）；
  4. 过滤掉 schema 未声明的多余参数再执行 `tool.fn(**filtered)`；
  5. `except Exception` 一律包装成 `{"error": f"{类型名}: {信息}"}` 的 JSON 字符串——**工具崩溃降级为观察结果**，模型看到错误还能换个姿势重试；
  6. `_finish()` 统一走 `tracer.log_tool`（名称/参数/成败/毫秒延迟），可观测是一等公民。
- `subset(allowed)` 白名单视图：共享同一批 ToolDef，只暴露允许的子集——第6期子 Agent 工具隔离的基础，且视图调不到的工具返回"未知工具"。
- `build_tool_registry(deps: ToolDeps)` 工厂装配 9 个工具，`ToolDeps`（tracer / retriever / memory_manager / skill_manager）全部依赖注入，None 表示能力未启用、对应工具返回明确错误——没有全局单例。

### 3. 9 个工具清单（含医疗特有设计）

| 工具 | 用途 | 参数要点 | 医疗特有设计 |
|---|---|---|---|
| `query_appointment` | 查预约单详情 | `appointment_id` 必填，如 GH-2026-001 | 只读；查不到时提示单号格式 |
| `cancel_appointment` | 取消/改期预约 | `action` 枚举 `cancel`/`reschedule`；改期需 `new_date` | **写操作**：真实修改 mock 状态；`cancelled`/`completed` 状态拒绝再操作；取消含退费话术 |
| `query_medicine` | 查药品说明 | `name`，如 布洛芬 | 目录仅 8 种教学药品，未收录时列出全部收录药 |
| `check_drug_interaction` | 查两药联用风险 | `drug_a`、`drug_b` | **"未收录"≠"无风险"**：无记录时返回 `severity="未收录"` 且 advice 明说"不等于无风险"；任一药品不在目录则直接失败 |
| `query_lab_report` | 查检验报告 | `report_id`，如 LAB-2026-001 | 顶层 `has_critical` 布尔字段（任一指标 flag=="危急"），供第10期危急值升级直接判定 |
| `query_department` | 按症状推荐科室 | `keyword` | **三级匹配**：①关键词整串命中科室信息 ②科室名/症状词出现在关键词里 ③无命中→显式失败并列出全部科室，绝不猜一个 |
| `search_knowledge` | 检索知识库（第5期） | `query` 必填，`top_k` 默认 3 | 返回带 `source=文档#小节` 的片段，且经过 `sanitized` 清洗（数据/指令分离） |
| `recall_user_memory` | 查用户健康档案（第7期） | `query` 可选 | 用药咨询前先查过敏史/用药史 |
| `load_skill` | 加载技能流程（第8期） | `skill_name` | 匹配技能场景（如 triage 分诊）时先取操作流程 |

其中后三个用闭包把 `deps.retriever / deps.memory_manager / deps.skill_manager` 捕获进 lambda——依赖注入在工厂里完成，工具函数本身保持纯参数签名。

### 4. mock 数据的"状态成对覆盖"（app/agent/tools/mock_data.py）

设计原则（沿用 参照项目 思路）：**每条工具分支都要有可触发的用例**——

- 预约覆盖 `booked / pending_payment / completed / cancelled` 四种状态；
- 药品覆盖 OTC 与处方药两类；
- 相互作用覆盖 高/中/低 风险三档，外加"两药都在目录但组合未收录"（氯雷他定+阿莫西林）；
- 报告覆盖 正常 / 偏高 / 危急（LAB-2026-004 血钾 2.6，flag="危急"）。

可变状态管理：只有预约是可写的——`_APPOINTMENTS_SEED` 是种子，`APPOINTMENTS` 是运行时副本，`reset_mock_data()` 用 `copy.deepcopy` 深拷贝恢复（模块导入时即执行一次）。`tests/test_tools.py` 的 `fresh_data` autouse fixture 在每个用例前后各调一次 `reset_mock_data()`，保证"取消 GH-2026-001"这类写操作不在测试间串扰。

### 6. 消息协议细节：tool 消息不能成为"孤儿"

循环往 `raw_messages` 里写三种消息，顺序有讲究：

1. `{"role": "assistant", "content": ..., "tool_calls": [...]}`——每个 tool_call 带 `id`，参数经 `_dump_args` 序列化（序列化失败兜底 `"{}"`，不让一条坏参数炸掉整轮）；
2. `{"role": "tool", "tool_call_id": tc["id"], "content": out}`——**id 必须与请求一一配对**，OpenAI 兼容接口会校验：tool 消息没有对应的 tool_calls 就是"孤儿"，请求直接被 4xx 拒绝；
3. 收尾的 `{"role": "assistant", "content": 最终文本}`。

这个约束还影响了第2期的历史压缩：`compress_history()` 回退切分点时专门跳过 tool 消息开头（`while ... role == "tool": split -= 1`），就是为了不从 tool 消息中间切开历史、制造孤儿。模型每一轮看到的上下文由 `render_messages(system_content)` 组装：system → 记忆区块（若有）→ 摘要（若有）→ raw_messages——ReAct 的"记忆"就是这份不断变长的消息列表本身。

### 7. 主流程挂接（app/agent/chat.py）

`MedicalAgent.chat()` 的完整流水线：`handle_redflag`（规则短路）→ 入历史 → 组装 system（SYSTEM_PROMPT + 技能目录）→ `react(system, self.tools)` → `safety_post_process`（注入扫描/引用校验/危急值升级，第10期）→ `extract_structured`（文本 → MedicalResponse）→ `finish_turn`（记忆更新/压缩/落盘）。工具层只服务于 react 这一段，其余环节与工具无关。

## 知识点与面试考点

1. **ReAct 与普通 function calling 调用的区别？** 要点：单次 function calling 是"一问一答一调用"；ReAct 是循环——观察结果回填上下文后继续推理，可串联多工具（先 `load_skill` 再 `query_department`），直到模型自行判断信息足够。
2. **为什么工具异常要包装成 JSON 观察结果？** 要点：异常一旦穿透到循环层，整个回合失败；包装成 `{"error": ...}` 后模型能读到失败原因并自我纠正（换参数、换工具、直接道歉说明）。"错误也是观测"。
3. **步数上限耗尽后为什么"去掉工具再问一次"？** 要点：硬截断会让用户拿到半截话；去掉 tools 强制模型只用已有观察收尾，保证每回合都有完整最终回复。
4. **`integer` 类型校验为什么要排除 bool？** 要点：Python 中 `isinstance(True, int)` 为 True，不排除的话 `"top_k": true` 能混过校验。
5. **检索型工具的"不知道"为什么要显式表达？** 要点：`check_drug_interaction` 未收录时说"不等于无风险"、`query_department` 匹配不到时明确失败——否则模型会把"查无记录"脑补成"安全/没问题"，这在医疗场景是事故。
6. **写操作工具如何做测试隔离？** 要点：种子数据与运行时副本分离 + `reset_mock_data()` 深拷贝重置 + autouse fixture，写操作改脏的状态不会泄漏到下一个用例。
7. **subset 白名单与"删掉工具"有何不同？** 要点：subset 返回共享 ToolDef 的新注册器视图，原注册表不受影响；被排除的工具在视图里按"未知工具"报错，权限边界清晰且零拷贝。

## 与 参照的电商客服教学项目 对照与改进

| 维度 | 参照项目（app/agent/tools/registry.py 等） | med 本项目 |
|---|---|---|
| schema 来源 | `TOOL_DEFINITIONS` 列表与 `_TOOL_MAP` 字典两处维护 | `ToolDef` 四元组，schema 与实现绑定，单一来源 |
| 执行入口 | `execute_tool()`：查表即调，无参数校验 | `execute()`：必填/类型校验 → 过滤未声明参数 → 异常包装 |
| 可观测 | 无 tracer 埋点 | `_finish()` 统一 `log_tool`（成败/延迟） |
| 白名单 | `ToolManager._filter_tools` 过滤定义列表 | `ToolRegistry.subset()` 视图，ToolDef 共享 |
| 失败语义 | `{"error": "未知工具: ...}` | 错误信息附带可用工具清单，利于模型自纠 |
| ReAct 循环 | 在 Agent 类内各自实现一份 | `BaseAgentRuntime.react()` 全系统唯一，SubAgent 复用 |
| 场景安全 | `apply_refund` 提示"调用前应与用户确认" | 危急值 `has_critical` 字段、"未收录≠无风险"、导诊三级匹配显式失败 |

## 动手练习

1. **验证步数护栏**：把 `Settings(max_react_steps=1)` 传给 `MedicalAgent`，问一个需要串联两个工具的问题（如"咳嗽两周了挂什么科"），观察最终回复是否仍是完整句子（强制收尾生效），再对比 `max_react_steps=6` 时 `agent.tracer.tool_calls` 的差别。
2. **新增一个只读工具**：仿照 `query_lab_report` 写一个 `query_department_detail(department_name)`（读 `DEPARTMENTS`），在 `build_tool_registry` 注册，跑 `pytest tests/test_tools.py`，然后故意把参数类型传错，确认拿到的是 JSON 观察结果而非堆栈。
3. **体会"未收录≠无风险"**：分别问"华法林和阿司匹林能一起吃吗"、"氯雷他定和阿莫西林呢"，对比两个 `check_drug_interaction` 返回的 `severity` 与 `advice`；再思考若把"未收录"分支改成返回 `severity="低风险"` 会引入什么风险。
