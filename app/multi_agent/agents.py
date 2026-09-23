"""子 Agent 配置（第6期）：名字/提示词/工具白名单。

权限最小化的落地形态：每个子 Agent 只注册白名单内的工具
（registry.subset 生成视图），prompt 里再声明一遍 —— 双层约束。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.prompts.agents import (APPOINTMENT_PROMPT, EMERGENCY_PROMPT,
                                MEDICATION_PROMPT, REPORT_PROMPT)


@dataclass(frozen=True)
class SubAgentSpec:
    key: str
    name: str
    prompt: str
    tools: frozenset[str] = field(default_factory=frozenset)


SUBAGENT_SPECS: dict[str, SubAgentSpec] = {
    "appointment": SubAgentSpec(
        key="appointment", name="挂号分诊助理", prompt=APPOINTMENT_PROMPT,
        tools=frozenset({"query_appointment", "cancel_appointment",
                         "query_department", "search_knowledge",
                         "recall_user_memory", "load_skill"})),
    "medication": SubAgentSpec(
        key="medication", name="用药咨询助理", prompt=MEDICATION_PROMPT,
        tools=frozenset({"query_medicine", "check_drug_interaction",
                         "recall_user_memory", "search_knowledge"})),
    "report": SubAgentSpec(
        key="report", name="报告解读助理", prompt=REPORT_PROMPT,
        tools=frozenset({"query_lab_report", "query_department",
                         "search_knowledge", "load_skill"})),
    "emergency": SubAgentSpec(
        key="emergency", name="急诊通道助理", prompt=EMERGENCY_PROMPT,
        tools=frozenset()),  # 急症：不配工具，强制走应答模板式快速回复
}
