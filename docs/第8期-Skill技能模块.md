# 第8期 · Skill 技能模块：SKILL.md 与渐进式披露

> 把"标准操作流程"写成一个个技能文件：启动时只注入目录（约 100 token/技能），模型命中场景再按需加载全文——指令不再撑爆上下文，也不会短到执行走样。

## 本期目标

- 理解 Agent Skills 开放标准的形式：目录 + SKILL.md + frontmatter；
- 掌握渐进式披露两阶段：目录注入 vs load_skill 按需加载全文；
- 读懂三个医疗技能的流程要点（红线前置/危急值置顶/写操作必须确认）；
- 说清 frontmatter 为什么用手写正则解析而不引 PyYAML。

## 架构与数据流

```text
启动阶段（SkillManager.__init__ → _discover）
┌────────────────────────────────────────────────┐
│ app/agent/skills/definitions/                  │
│ ├── triage/SKILL.md            （分诊）        │
│ ├── lab-report/SKILL.md        （报告解读）    │
│ └── appointment-change/SKILL.md（预约退改）    │
└───────────────┬────────────────────────────────┘
                ▼ 只解析 frontmatter（name + description）
        SkillMeta（body 未加载，惰性）
                ▼
   build_catalog_prompt() ≈ 每技能一行，注入 system prompt
   （编排器：system = spec.prompt + catalog）

运行阶段（ReAct 循环内，按需）
模型读目录 → 判断命中技能 → 调用 load_skill(skill_name)
                ▼
   SkillManager.load_skill → SkillMeta.load_body()   ← 首次触发才读文件
                ▼
   {"success": True, "instructions": 完整流程} 作为工具结果进入上下文
                ▼
   模型按流程逐步执行（调用 query_department / query_lab_report /
   cancel_appointment 等已有工具完成动作）
```

## 核心实现讲解

### app/agent/skills/loader.py —— 发现与渐进式披露

`SkillManager(skills_dir, enabled)` 启动时 `_discover()` 扫描
definitions 下每个含 SKILL.md 的子目录，只解析 frontmatter 的
name/description 存进 `SkillMeta`（dataclass：name/description/path/body/
_loaded）。两个关键方法：

- **build_catalog_prompt()**：生成"## 可用技能（Skills）"段落——每个技能
   一行 `**名字**：描述`，外加一句"命中时先 load_skill 再按流程执行，
   不匹配不要强行套用"。三个技能总共几百字符
   （`test_catalog_prompt_is_compact` 断言 < 1200），这就是**第一阶段**。
- **load_skill(skill_name) → dict**：查不到返回 `{"success": False,
   "error": "未找到技能…可用：[...]"}`；查到则触发 `SkillMeta.load_body()`
   ——首次调用才 `read_text` 读全文并缓存（`_loaded` 标志），这是
   **第二阶段**。

这套两阶段就是"渐进式披露"（progressive disclosure）：上下文里常驻的
只有目录（每技能约 100 token），完整流程（几百到上千 token）只在命中
场景时进入一次。它同时解决一对矛盾：**指令写全了，常驻 system prompt
占爆上下文、还稀释模型对安全边界的注意力；指令写短了，执行走样**。
目录管"知道有哪些流程"，全文管"按流程做"。

### frontmatter 解析：手写正则而非 PyYAML

`_parse_frontmatter` 用 `re.match(r"^---\s*\n(.*?)\n---", content,
re.DOTALL)` 切出 frontmatter 块，再逐行 `partition(":")` 取键值
（跳过以空格/`-`/`#` 开头的行，避免误吃列表与注释）；`_parse_body`
把 `---` 块之后的内容剥出来。不引 PyYAML 的理由：

1. 技能元数据只需要**扁平的 key: value 字符串**，没有嵌套/列表/多行
   标量的需求——YAML 的表达力在这里是闲置的；
2. 少一个三方依赖，安装更轻（教学项目对环境敏感），也避开某些平台
   上 C 扩展编译的坑；
3. 失败模式温和：解析不出 name/description 的目录在 `_discover` 里
   直接跳过，不会带病启动。

代价是解析器只认自己约定的子集——技能作者若在 frontmatter 里写
嵌套结构会被静默忽略，需要用 `test_frontmatter_parsed` 这类用例守住。

### app/agent/tools/skill_tool.py —— 执行入口

`load_skill(skill_manager, skill_name)` 与记忆工具同款写法：第一个参数
是注入的依赖，None 时返回"技能系统未启用"。注册发生在
`build_tool_registry`（app/agent/tools/registry.py）：
`lambda skill_name: load_skill(deps.skill_manager, skill_name)`。
注意白名单联动：SUBAGENT_SPECS 里只有 appointment 与 report 的工具集
含 load_skill，medication 没有（其场景无技能可套），emergency 零工具
——技能的加载权同样受工具白名单管。

### 三个技能的流程要点（definitions/*/SKILL.md）

- **triage（分诊）**：第 1 步就是**急症红线检查前置**——命中胸痛/呼吸
  困难/大出血/意识不清等，立即停止常规分诊转急诊；随后一次最多追问
  两个问题、query_department 匹配科室（匹配不到不猜）、给出建议并固定
  附"以医生意见为准"。禁止事项：不下诊断；14 岁以下一律儿科；
  孕期出血按急症处理。
- **lab-report（报告解读）**：先 query_lab_report 拿真实报告（没有报告号
  先问，**禁止凭口述数值解读**）；检查 `has_critical` 字段——为 true 时
  **危急值提示置顶**并升级 urgency/requires_human，再做分级解读
  （指标→偏离方向→常见原因→建议）；正常指标一句带过。禁止：不下诊断、
  不解读自带数值、多项危急时不再逐条分析直接转急诊。
- **appointment-change（预约退改）**：定位预约（query_appointment）→
  核验状态（completed/cancelled 不可操作）→ **写操作前必须复述并等用户
  确认**（给出话术示例）→ 执行 cancel_appointment（cancel/reschedule）
  → 失败如实转达。禁止：未确认就写、代猜单号/日期、费用争议转人工。

三个技能共同的骨架：**安全检查前置 + 分步操作 + 禁止事项兜底**。
技能是纯 prompt 资产（不写代码），改流程 = 改 Markdown，重启即生效。

### 编排器/单 Agent 的接入点

`MultiAgentOrchestrator.chat()` 与 `MedicalAgent.chat()` 都是同一句：
`system = spec.prompt + self.skill_manager.build_catalog_prompt()`
（skills_enabled 时）。目录对所有子 Agent 可见，加载权则由各白名单
里的 load_skill 决定。

## 知识点与面试考点

1. **Agent Skills 的标准形式是什么？**
   一个技能 = 一个目录 + SKILL.md；文件头部 YAML frontmatter 声明
   name/description（供发现与路由），正文 Markdown 是给模型执行的
   操作指令。内容与代码解耦，非程序员也能维护流程。
2. **渐进式披露的两个阶段分别解决什么？**
   阶段一目录常驻 system prompt（低成本"知道有什么"）；阶段二
   load_skill 按需加载全文（高保真"怎么做"）。没有阶段二，指令写全
   会占爆上下文；没有阶段一，模型根本不知道该加载什么。
3. **技能和工具有什么区别？**
   工具是可执行的动作（函数 + JSON Schema 参数）；技能是"何时、
   以什么顺序、带着什么安全约束去组合工具"的流程知识（纯文本）。
   技能驱动工具，不是替代工具。
4. **为什么 SkillMeta 要惰性加载 body？**
   启动只需 name/description 做目录；多数会话只会命中 0~1 个技能，
   提前读全文是纯浪费（`_loaded` 标志保证每个技能至多读一次盘）。
5. **load_skill 失败时返回什么？为什么列出可用技能名？**
   `{"success": False, "error": "未找到技能…可用：…"}`——把正确选项
   回给模型，让它有机会自我纠正（拼错名字时重试），符合工具输出
   "永不抛异常、给可行动信息"的原则。
6. **SKILL.md 的 description 应该怎么写？**
   它是路由依据：写清触发场景与关键词（如"用户要退货/换货时使用"），
   覆盖过宽会误触发，过窄会漏。med 的三个 description 都点名了场景。
7. **为什么不在 frontmatter 里用复杂 YAML 特性？**
   解析器只支持扁平 key: value；复杂结构会被静默忽略。约定优于配置：
   技能作者遵循子集，测试守住解析正确性。
8. **技能指令如何防止被工具结果"顶掉"？**
   全文以 tool 消息进入对话，位置在 system 之后；关键约束（如"必须
   确认"）在技能里重复强调，且写操作的实际执行仍受工具层参数校验与
   prompt 安全边界双重约束——技能是流程约束，不是唯一防线。

## 与 参照的电商客服教学项目 对照与改进

| 维度 | 参照项目（app/agent/skills/） | med（本项目） |
|---|---|---|
| 标准形式 | 相同：definitions/ 下三个技能（退货/订单跟踪/商品推荐三个目录），SKILL.md + frontmatter | 相同形式，三个医疗技能（triage/lab-report/appointment-change） |
| 目录 prompt | build_catalog_prompt 较长：额外附 4 步"技能使用方式"说明 | 更紧凑（<1200 字符，有测试约束），说明压缩为一句话 |
| frontmatter 解析 | 正则 + 多行值拼接（_parse_frontmatter 较复杂） | 更简单的扁平 key: value 解析，跳过列表/注释行 |
| 依赖注入 | skill_tool 用 set_skill_manager() 全局单例 | load_skill(skill_manager, ...) 构造注入，闭包捕获 |
| 技能内容 | 电商流程（退货/推荐/查单），安全属性弱 | 每个技能内嵌安全关键步骤：红线前置、危急值置顶、写操作必须确认 |
| 加载权限 | 三个子 Agent 白名单均含 load_skill | 白名单差异化：appointment/report 有 load_skill，medication/emergency 无 |
| 失败反馈 | 未找到时列出可用技能 | 相同（这处 参照项目 做对了，予以保留） |

## 动手练习

1. **新增 medication-guide 技能**：为用药咨询场景写
   definitions/medication-guide/SKILL.md（frontmatter + 分步流程 +
   禁止事项，如"联用必须核查、未收录≠无风险"），把 load_skill 加进
   SUBAGENT_SPECS["medication"] 的白名单，并在 tests/test_skills.py
   补发现与加载用例。
2. **量化渐进式披露的收益**：统计三个 SKILL.md 全文总字符数，与
   build_catalog_prompt() 的长度对比（可用 len() 直接算），估算每次
   会话省下的 token；再写一个假 client 故意每轮都 load_skill 三个技能，
   观察 raw_messages 的膨胀速度。
3. **体验"写操作必须确认"**：用 `python main.py` 走一遍"取消预约
   GH-2026-001"，在确认话术出现前回答"再想想"，观察 agent 是否真的
   没有调用 cancel_appointment（看 tracer 的工具调用记录）——这就是
   技能流程约束在起作用。
