"""结构化输出 schema（Pydantic v2）。

面试考点：结构化输出 = 让自由文本在系统边界处收敛为可校验的对象。
intent/urgency 用受控枚举（模型只能从白名单里选），confidence 约束在 [0,1]，
requires_human 是人机协作的总开关 —— 医疗场景里"转人工"必须是确定性字段而非话术。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    greeting = "greeting"              # 问候/寒暄
    appointment = "appointment"        # 挂号/预约/退改
    medication = "medication"          # 用药咨询
    lab_report = "lab_report"          # 检验报告解读
    policy = "policy"                  # 就诊政策/流程/医保
    emergency = "emergency"            # 急症红线
    other = "other"


class UrgencyLevel(str, Enum):
    routine = "routine"      # 日常咨询
    attention = "attention"  # 需留意，建议尽快就诊
    urgent = "urgent"        # 紧急，当天就诊
    emergency = "emergency"  # 急症红线，立即急诊/120


class MedicalResponse(BaseModel):
    """每轮对话的最终结构化产出。"""

    intent: IntentType = Field(description="意图分类", default=IntentType.other)
    confidence: float = Field(ge=0.0, le=1.0, description="置信度", default=0.8)
    reply: str = Field(min_length=1, description="给用户的回复正文")
    requires_human: bool = Field(default=False, description="是否需要转人工客服/医生")
    urgency: UrgencyLevel = Field(default=UrgencyLevel.routine, description="紧急程度")
    follow_up_question: Optional[str] = Field(default=None, description="可选的追问")


INTENT_LABELS = {
    IntentType.greeting: "问候",
    IntentType.appointment: "挂号预约",
    IntentType.medication: "用药咨询",
    IntentType.lab_report: "报告解读",
    IntentType.policy: "政策咨询",
    IntentType.emergency: "急症",
    IntentType.other: "其他",
}
