"""评估数据集（第9期）：EvalCase 结构 + 加载。

核心语义（与 ecom 一致）：期望项为 None/空 表示"本条用例不考察该维度"，
聚合时跳过而不是记 0 分 —— 否则没有 expected_tools 的问候用例会拖垮均值。
safety_critical=True 的用例是安全红线用例：requires_human/urgency 必须达标。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalCase:
    id: str
    description: str = ""
    turns: list[str] = field(default_factory=list)
    # 结果期望
    expected_intent: Optional[str] = None
    expected_keywords: list[str] = field(default_factory=list)
    expected_requires_human: Optional[bool] = None
    expected_urgency: Optional[str] = None
    # 过程期望
    expected_tools: list[str] = field(default_factory=list)
    min_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    expected_route: Optional[str] = None
    # 安全
    safety_critical: bool = False


def load_dataset(path: str) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for d in data.get("cases", []):
        cases.append(EvalCase(
            id=d["id"], description=d.get("description", ""),
            turns=d.get("turns", []),
            expected_intent=d.get("expected_intent"),
            expected_keywords=d.get("expected_keywords", []),
            expected_requires_human=d.get("expected_requires_human"),
            expected_urgency=d.get("expected_urgency"),
            expected_tools=d.get("expected_tools", []),
            min_tool_calls=d.get("min_tool_calls"),
            max_tokens=d.get("max_tokens"),
            expected_route=d.get("expected_route"),
            safety_critical=bool(d.get("safety_critical", False))))
    return cases
