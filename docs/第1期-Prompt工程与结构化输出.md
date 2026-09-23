# 第1期 · Prompt 工程与结构化输出

> 用一个受控的 MedicalResponse 收敛模型输出，用带重试退避、可降级、可离线的 LLM 客户端承载它。

## 本期目标

- 说清 `MedicalResponse` 每个字段为什么这样设计（枚举、区间、确定性开关）。
- 拆解 `app/prompts/consultant.py` 的 SYSTEM_PROMPT：身份句 + 角色定位 / 安全边界 / 回复规范 / 长期能力四段。
- 掌握结构化输出双路：`response_format` 优先，失败降级为 prompt 引导 JSON + 剥围栏 + pydantic 校验。
- 理解 `with_retry` / `_is_transient` 的指数退避重试：只重试瞬态错误，`sleeper` 可注入所以纯函数可测。
- 理解 `MockLLMClient` 为什么"会读对话历史"才让离线测试有意义。

## 架构与数据流

```text
用户输入
   │
   ▼
MedicalAgent.chat() ──▶ ReAct 循环（app/agent/base.py react）
   │                      │ client.chat(messages, tools, purpose="react")
   │                      ▼
   │         ┌────────────────────────────────────┐
   │         │ app/llm/client.py                  │
   │         │  with_retry(指数退避 1s/2s/4s)      │ ◀─ 只重试 429/5xx/超时/网络
   │         │  build_client：is_offline ?        │
   │         │    MockLLMClient（读对话历史）      │    离线
   │         │    OpenAICompatClient              │    在线
   │         └────────────────────────────────────┘
   ▼
最终文本 ──▶ extract_structured（app/agent/base.py）
   │           client.parse_structured(text, MedicalResponse)
   │             ① response_format=schema（beta 接口，temperature=0.0）
   │             ② 任何异常 → _parse_fallback：schema 写进 prompt
   │                + _strip_fences 剥围栏 + model_validate_json
   ▼
MedicalResponse（pydantic 校验通过的意图/紧急度/转人工开关…）
```

## 核心实现讲解

### `app/schemas/response.py`：字段即约束

- **`intent: IntentType`**——七值受控枚举（greeting / appointment / medication / lab_report / policy / emergency / other）。模型只能从白名单里选，路由与统计才有可靠键；自由文本分类迟早漂移。`INTENT_LABELS` 提供中文标签，`test_intent_labels_cover_all` 保证标签与枚举同步。
- **`urgency: UrgencyLevel`**——四级（routine / attention / urgent / emergency），把"多急"变成可分支的信号（急症红线直接给急诊应答）。
- **`confidence: float = Field(ge=0.0, le=1.0)`**——区间校验，越界直接 ValidationError（`test_confidence_bounds`）。
- **`reply: str = Field(min_length=1)`**——给用户的正文，必须非空（`test_reply_required`）。
- **`requires_human: bool`**——人机协作总开关。医疗场景"转人工（医生/药师）"必须是**确定性字段**而非模型话术：红旗短路（`app/agent/base.py` 的 `handle_redflag`）构造响应时直接置 True，安全后处理（`safety_post_process` 返回 `forced_human`，由 `app/agent/chat.py` 应用为 `result.requires_human = True`）同样能规则强制，下游读 bool 即可决策。
- **`follow_up_question: Optional[str]`**——可选追问，为多轮引导留位。

### `app/prompts/consultant.py`：SYSTEM_PROMPT 的分段

一句身份定位（"你是「小医」，一家互联网医院的健康咨询与导诊助手"）+ 四个 `##` 段：

- **角色定位**：挂号/退改、科室建议（导诊，非诊断）、用药咨询、报告解读、政策答疑；以及工具原则——"必须先调用工具获取真实数据，再回答；禁止凭记忆编造"。
- **安全边界（硬约束，优先级最高）**：不诊断不开处方不荐剂量；急症红线（胸痛、呼吸困难、大出血等）立即 120/急诊，**不做常规查询**；健康建议附免责声明；用户坚持要诊断/开药时拒绝并 `requires_human` 转人工。边界写进 system prompt 而不是靠模型自觉。
- **回复规范**：中文口语化 3~6 句、先结论后依据、引用知识库注明来源（如「根据【就诊指南#挂号流程】…）、以一句追问收尾。
- **长期能力**：预告系统会注入"用户健康档案"（用药前核对过敏与联用风险）与"可用技能"目录（匹配场景先 `load_skill`）——为记忆与技能模块留好接口。

### `app/llm/client.py`：结构化输出双路

`OpenAICompatClient.parse_structured(messages, schema)`：

- **主路**：`_call(lambda: self._inner.beta.chat.completions.create(..., temperature=0.0, response_format=schema))`，取 `resp.choices[0].message.parsed`，为 None 抛 `LLMError`。抽取任务用 temperature=0 求确定性。
- **降级 `_parse_fallback`**：任何异常都落进来（接口不支持 response_format、网络抖动重试后仍失败等）。把 `schema.model_json_schema()` 整个 JSON 序列化进 system prompt（"只输出一个 JSON 对象（禁止代码块）"），只取 `messages[-1]` 的内容拼成新会话再请求一次；返回文本先过 `_strip_fences`（正则剥 ```` ```json ```` 围栏，失败再截取首个 `{` 到最后一个 `}`），然后 `schema.model_validate_json`，ValidationError 包装成 `LLMError`。
- **schema 自动同步**是关键改进：参照项目的降级是手写 JSON 模板（枚举值拼字符串），pydantic 改了模板不会跟着改；这里 prompt 由 `model_json_schema()` 现场生成，永不漂移。这段逻辑放在客户端而不是 Agent 类里，Mock 客户端实现同一个抽象方法即可整体复用。

### `app/llm/client.py`：重试退避

- `_is_transient(exc)` 三层判定：`TransientLLMError`；openai 的 `RateLimitError / APITimeoutError / APIConnectionError / InternalServerError`（import 失败容错，不强依赖 openai 包）；兜底看 `status_code in (408, 409, 429, 500, 502, 503, 504)`。**4xx 参数错误不在此列——重试只会再失败一次，立刻抛。**
- `with_retry(fn, max_retries=3, base_delay=1.0, sleeper=time.sleep, is_transient=_is_transient)`：指数退避 `base_delay * 2**attempt`，且**最后一次尝试失败后不再睡眠直接抛错**——`max_retries=3` 时实际睡眠序列是 `[1.0, 2.0]`（`tests/test_llm.py::test_retries_transient_then_succeeds` 正是这样断言的），耗尽后抛 `LLMError`。`sleeper` 与 `is_transient` 都是参数，测试里传 `delays.append` 就能验证退避节奏，不需要真睡眠。
- 客户端里的接法：`OpenAICompatClient._call` 先用 `guarded` 把瞬态异常包装成 `TransientLLMError` 再交给 `with_retry`，真实调用方只见到统一的 `LLMError`。

### `app/llm/client.py`：MockLLMClient 为什么必须"会读对话历史"

- 与真实客户端实现同一个 `BaseLLMClient` 抽象（`chat` / `parse_structured`），`build_client` 按离线判定切换——调用方无感。
- `chat` 按 `purpose` 分派：`router` 走 `_route`（关键词路由 emergency/report/medication/appointment），`summarize` 返回"（摘要）"+用户消息前 150 字，`react` / `react:*` 走 `_react`。
- `_react` 用 `self._counts` 以 `(hash(系统提示)%10**8, user[:120])` 为 key 计步：第 n 次调用返回 `_plan(user)` 的第 n 个工具调用，计划走完才出最终文本——模拟真实的多步 ReAct。
- `_plan(user)` 用正则/关键词产出工具序列：`GH-\d{4}-\d{3}` 查预约（含"取消/退"再追加 `cancel_appointment`）、两个药名或联用词触发 `check_drug_interaction`、单药名 `query_medicine`、政策词 `search_knowledge`、`LAB-\d{4}-\d{3}` 查报告、`load_skill("triage")` 等。
- **关键在 `_final_text`**：调 `_tool_results(messages, since_last_user=True)` 从最近一条用户消息之后收集 `(工具名, 结果dict)`，再从工具结果 JSON 里**提取字段**拼回复（预约的科室/医生/时间/状态、相互作用等级、报告异常指标……），工具失败的 `error` 如实转达，结尾统一免责声明。`since_last_user=True` 保证多轮会话里历史轮次的工具结果不会被误当成本轮素材。
- 因为 Mock 真的消化工具结果，离线跑出的轨迹与真实模式**结构一致**——`test_react_two_step_with_tool_result_echo` 断言最终回复里出现工具返回的"呼吸内科""陈志远"。测试、评估、演示因此可信，而不是走个过场。
- `parse_structured` 的 Mock 实现按 `schema.model_fields` 分派：有 `intent` + `reply` 字段（即 `MedicalResponse`）走 `_medical_fields(text)`（关键词判定 intent/urgency/requires_human，"急诊/120" → emergency 且转人工，"危急/高风险/异常" 把 routine 升到 attention）；有 `interaction_summary`（长期记忆提取）或有 `facts`（短期记忆提取）返回空对象——下游拿到的是**通过同一套 pydantic 校验的实例**，不是绕过 schema 的裸 dict。

### `tests/test_llm.py`：怎么测这一层

- `TestRetry` 四连：瞬态错误重试两次后成功且退避序列 `[1.0, 2.0]`；4xx 参数错误只调一次立刻抛；重试耗尽抛 `LLMError`；`_is_transient` 按 `status_code` 分类（429/503 可重试，400/401 不可）。全部通过注入 `sleeper` 完成，零真实等待。
- `TestMockClient` 覆盖：router 分类（含默认兜底 appointment）、两步 ReAct（工具调用 → 工具结果回填 → 最终文本引用工具字段）、问候语不调工具、`purpose="react:medication"` 子代理前缀、`parse_structured` 的三种 schema 分派（含急症识别出 `UrgencyLevel.emergency` 且 `requires_human is True`）。
- `test_offline_factory_builds_mock`：`build_client(Settings(mock_mode=True), Tracer())` 返回 `MockLLMClient`——工厂的切换行为本身也被钉死。

### `build_client`：工厂与离线切换

```python
def build_client(settings: Settings, tracer: Tracer) -> BaseLLMClient:
    if is_offline(settings):
        return MockLLMClient(tracer=tracer)
    return OpenAICompatClient(settings, tracer)
```

- 切换点全项目只此一处，且**返回的是抽象类型 `BaseLLMClient`**：调用方（`BaseAgentRuntime`、`MemoryManager`、summarizer）面向接口编程，根本不知道自己拿到的是真是假。
- `OpenAICompatClient.__init__(settings, tracer, inner_client=None)` 的第三个参数是测试后门：不传就 `import openai` 现建客户端；传了就用你给的 fake——降级路径、tool_calls 解析、usage 容错都能在无网络环境下单测。
- `chat` 里对 `tool_calls` 参数做 `json.loads`，失败时落回 `{}`；`usage` 用 `getattr(..., 0)` 容错——兼容端点的返回字段参差是常态，边界处兜住比在业务层到处判空干净。

## 知识点与面试考点

- **为什么 intent/urgency 用枚举？** 输出受控白名单才能做路由、统计、阈值告警；字符串输出无法穷举分支。
- **requires_human 为什么是 bool 字段？** 转人工是工程决策不是话术；确定性字段可被规则强制（红旗短路、危急值升级），下游零解析成本。
- **response_format 与 prompt 引导 JSON 各自的风险？** 前者依赖接口支持（部分兼容端点不支持 beta 接口）；后者依赖模型听话，所以要剥围栏 + pydantic 校验兜底，双路互补。
- **`_strip_fences` 为什么要两步？** 先正则剥 ```` ```json ```` 围栏，失败再截取首尾大括号——模型常在 JSON 外加说明文字，两步能覆盖 majority 情况。
- **为什么 4xx 不重试？** 参数/鉴权错误是确定性失败，重试浪费配额还可能触发限流；只有 429/5xx/超时/网络这类瞬态错误值得退避重试。
- **指数退避公式与注入点？** `base_delay * 2**attempt`；`sleeper` 注入让测试断言退避序列 `[1.0, 2.0]` 而不必真睡，`is_transient` 注入可模拟任意错误分类。
- **Mock 客户端的底线是什么？** 与真实实现同接口、同消息协议（tool_calls/tool 消息都真实流动），否则离线测试验证的是假路径。
- **抽取为什么 temperature=0？** 分类/抽取要稳定可复现，高温只会让同一输入得到不同 intent。

## 与 参照的电商客服教学项目 对照与改进

| 方面 | 参照项目 做法 | 本项目做法与理由 |
| --- | --- | --- |
| 提取逻辑位置 | `参照项目的单Agent类._extract_structured_response/_fallback`，且 orchestrator 里复制了一份 | 下沉到 `OpenAICompatClient.parse_structured` / `MockLLMClient.parse_structured`，Agent 只调抽象接口，全系统一份实现 |
| 降级 prompt 的 schema | 手写 JSON 模板字符串，枚举值拼接，pydantic 改动会与模板漂移 | `schema.model_json_schema()` 现场生成进 prompt，schema 与 prompt 永远同步 |
| 围栏剥离 | `startswith("```")` 时 split/rsplit 手工剥 | `_strip_fences`：正则围栏 → 首尾大括号截取，双保险 |
| 重试 | 无，`create` 失败直接抛给上层 | `with_retry` 指数退避 + `_is_transient` 分类，`sleeper`/`is_transient` 可注入纯函数单测 |
| 错误语义 | 裸异常 | `LLMError` / `TransientLLMError`，调用方可分类处理 |
| 离线/Mock | 无 Mock 客户端，测试依赖真实 Key | `MockLLMClient` 同接口且读对话历史，离线轨迹可信 |
| token/延迟计量 | 评估时 monkey-patch 采集 | 每次 `_trace` → `tracer.log_llm(purpose=...)` 实时计量 |

## 动手练习

1. **扩展 Mock 的工具计划**：在 `_plan` 里加一条规则（例如用户提到"发烧/咳嗽"时计划 `load_skill("triage")`），并仿照 `tests/test_llm.py::TestMockClient` 补一条断言 `tool_calls[0]["name"]` 的测试。
2. **验证降级路径**：构造一个 fake `inner_client`，让 `beta.chat.completions.create` 抛异常、`chat.completions.create` 返回带 ```` ```json ```` 围栏的 JSON 文本，用 `OpenAICompatClient(settings, tracer, inner_client=fake)` 断言 `parse_structured` 仍返回合法 `MedicalResponse`（提示：降级返回的文本会经过 `_strip_fences`）。
3. **参数化边界测试**：把 `test_confidence_bounds` 改成 `pytest.mark.parametrize`，覆盖 -0.1 / 0 / 1 / 1.5 四个边界值。
