"""技能加载工具（第8期）：load_skill —— 渐进式披露的执行入口。"""

from __future__ import annotations

from typing import Optional

from app.agent.skills.loader import SkillManager


def load_skill(skill_manager: Optional[SkillManager], skill_name: str) -> dict:
    if skill_manager is None:
        return {"success": False, "error": "技能系统未启用"}
    return skill_manager.load_skill(skill_name)
