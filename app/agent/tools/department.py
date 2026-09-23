"""科室查询工具（第3期）：按症状/关键词推荐就诊科室（导诊，非诊断）。"""

from __future__ import annotations

from app.agent.tools.mock_data import DEPARTMENTS


def query_department(keyword: str) -> dict:
    """三级匹配：① 关键词整串命中科室信息 ② 科室名/症状词出现在关键词里
    ③ 无命中 —— 明确说"无法匹配"而不是猜一个（检索型工具的"不知道"
    必须显式，防止模型把未收录当成安全/正确）。"""
    for dep in DEPARTMENTS.values():
        haystack = dep["name"] + dep["description"] + "".join(dep["keywords"])
        if keyword and keyword in haystack:
            return {"success": True, "department": dep}
    for dep in DEPARTMENTS.values():
        if dep["name"] in keyword or any(
                k in keyword for k in dep["keywords"]):
            return {"success": True, "department": dep}
    return {"success": False,
            "error": f"无法根据「{keyword}」匹配科室，请描述主要症状后再试",
            "departments": list(DEPARTMENTS)}
