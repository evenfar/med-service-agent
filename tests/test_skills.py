"""技能模块测试（第8期）。"""

from __future__ import annotations

from app.agent.skills.loader import SkillManager


class TestLoader:
    def test_discovers_three_skills(self):
        sm = SkillManager()
        assert set(sm.skill_names) == {"triage", "lab-report",
                                       "appointment-change"}

    def test_frontmatter_parsed(self):
        sm = SkillManager()
        catalog = {s["name"]: s["description"] for s in sm.catalog()}
        assert "分诊" in catalog["triage"]
        assert "危急值" in catalog["lab-report"]

    def test_load_body_contains_steps(self):
        sm = SkillManager()
        body = sm.load_skill("triage")
        assert body["success"] and "急症红线" in body["instructions"]

    def test_load_unknown_skill(self):
        sm = SkillManager()
        out = sm.load_skill("不存在的技能")
        assert not out["success"] and "triage" in out["error"]

    def test_catalog_prompt_is_compact(self):
        sm = SkillManager()
        prompt = sm.build_catalog_prompt()
        assert "load_skill" in prompt and "triage" in prompt
        assert len(prompt) < 1200  # 渐进式披露：目录必须轻量

    def test_disabled_manager(self):
        sm = SkillManager(enabled=False)
        assert sm.build_catalog_prompt() == ""
        assert not sm.load_skill("triage")["success"]
