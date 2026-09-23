"""红旗症状检测（第10期）：急症红线在进 LLM 之前用确定性规则短路。

为什么用规则而不是让 LLM 判断：急症判定要的是零漏报 + 零延迟 + 可解释，
受控词表是这三点都能满足的方案；LLM 只作为第二路（structured output 的
urgency 字段）兜底模糊表达。这是"确定性工作不交给模型"原则的典型应用。
"""

from __future__ import annotations

import re

# 红旗症状/危急词表（教学版，生产应经医学审核并按科室分级）
RED_FLAG_PATTERNS: list[tuple[str, str]] = [
    (r"胸痛|心口疼|心前区.*疼", "胸痛"),
    (r"呼吸困难|喘不上气|气促|憋气", "呼吸困难"),
    (r"咯血|大出血|出血不止|血止不住", "出血"),
    (r"意识不清|昏迷|叫不醒|晕厥|晕倒", "意识障碍"),
    (r"抽搐|惊厥|癫痫发作", "抽搐"),
    (r"休克|血压测不到", "休克征象"),
    (r"剧烈头痛|炸裂样头痛", "剧烈头痛"),
    (r"药物过量|吃错药|误服", "药物过量"),
    (r"自杀|轻生|不想活", "心理危机"),
    (r"高热不退|40度以上|41度", "超高热"),
]

_EMERGENCY_REPLY_TEMPLATE = (
    "您描述的情况属于急症红线（{flags}），请立即拨打 120 或前往最近医院急诊科，"
    "不要等待线上回复。如身边有他人，请让他人陪同。"
)


def detect_red_flags(text: str) -> list[str]:
    """返回命中的红旗症状名列表（空列表=无）。"""
    hits: list[str] = []
    for pattern, name in RED_FLAG_PATTERNS:
        if re.search(pattern, text):
            hits.append(name)
    return hits


def emergency_reply(flags: list[str]) -> str:
    return _EMERGENCY_REPLY_TEMPLATE.format(flags="、".join(flags))
