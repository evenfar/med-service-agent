"""药品工具（第3期）：查药品说明 + 查药物相互作用（联用风险）。"""

from __future__ import annotations

from app.agent.tools.mock_data import DRUG_INTERACTIONS, MEDICINES


def query_medicine(name: str) -> dict:
    med = MEDICINES.get(name)
    if not med:
        known = "、".join(MEDICINES)
        return {"success": False,
                "error": f"药品「{name}」不在本目录中（教学目录），收录药品：{known}"}
    return {"success": True, "medicine": med}


def check_drug_interaction(drug_a: str, drug_b: str) -> dict:
    """查联用风险。按字母序对匹配；无记录时返回"未收录"而非"无风险"——
    教学要点：检索型工具的"不知道"必须显式表达，防止模型把未收录当成安全。"""
    pair = tuple(sorted([drug_a, drug_b]))
    record = DRUG_INTERACTIONS.get(pair)
    if record:
        return {"success": True, "drug_a": drug_a, "drug_b": drug_b,
                "severity": record["severity"], "effect": record["effect"],
                "advice": record["advice"]}
    if drug_a not in MEDICINES or drug_b not in MEDICINES:
        missing = [d for d in (drug_a, drug_b) if d not in MEDICINES]
        return {"success": False,
                "error": f"药品不在目录中：{'、'.join(missing)}，无法核查"}
    return {"success": True, "drug_a": drug_a, "drug_b": drug_b,
            "severity": "未收录", "effect": "教学相互作用表中无该组合的记录",
            "advice": "不等于无风险，联用前请咨询医生或药师"}
