"""技能加载器（第8期）：SKILL.md 发现 + 渐进式披露。

遵循 Anthropic Agent Skills 开放标准的形式：每个技能一个目录 + SKILL.md
（YAML frontmatter 声明 name/description，正文是操作指令）。
渐进式披露：启动时只把 name+description（约100 token/技能）注入 system prompt，
模型判断匹配后调用 load_skill 工具才加载全文 —— 解决"指令太长占爆上下文"
与"指令太短执行走样"的矛盾。frontmatter 用简化解析，不引入 PyYAML 依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillMeta:
    name: str
    description: str
    path: Path
    body: str = ""
    _loaded: bool = field(default=False, repr=False)

    def load_body(self) -> str:
        if not self._loaded:
            raw = self.path.read_text(encoding="utf-8")
            self.body = _parse_body(raw)
            self._loaded = True
        return self.body


def _parse_frontmatter(content: str) -> dict:
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return {}
    result: dict[str, str] = {}
    for line in m.group(1).strip().splitlines():
        if ":" in line and not line.startswith((" ", "-", "#")):
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def _parse_body(content: str) -> str:
    m = re.match(r"^---\s*\n.*?\n---\s*\n?", content, re.DOTALL)
    return content[m.end():].strip() if m else content.strip()


class SkillManager:
    def __init__(self, skills_dir: str = "app/agent/skills/definitions",
                 enabled: bool = True):
        self.skills_dir = Path(skills_dir)
        self.enabled = enabled
        self._skills: dict[str, SkillMeta] = {}
        if enabled:
            self._discover()

    def _discover(self) -> None:
        if not self.skills_dir.exists():
            return
        for d in sorted(self.skills_dir.iterdir()):
            skill_file = d / "SKILL.md"
            if not (d.is_dir() and skill_file.exists()):
                continue
            meta = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            name, desc = meta.get("name", ""), meta.get("description", "")
            if name and desc:
                self._skills[name] = SkillMeta(name, desc, skill_file)

    # ---------- 查询 ----------

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills)

    def catalog(self) -> list[dict]:
        return [{"name": s.name, "description": s.description}
                for s in self._skills.values()]

    def build_catalog_prompt(self) -> str:
        if not self.enabled or not self._skills:
            return ""
        lines = ["\n\n## 可用技能（Skills）",
                 "以下场景有标准操作流程。用户问题命中技能描述时，先调用 "
                 "`load_skill` 加载完整流程，再按流程执行："]
        for s in self._skills.values():
            lines.append(f"- **{s.name}**：{s.description}")
        lines.append("不匹配任何技能时正常回答即可，不要强行套用。")
        return "\n".join(lines)

    def load_skill(self, skill_name: str) -> dict:
        if not self.enabled:
            return {"success": False, "error": "技能系统未启用"}
        skill = self._skills.get(skill_name)
        if not skill:
            return {"success": False,
                    "error": f"未找到技能「{skill_name}」，可用：{self.skill_names}"}
        return {"success": True, "skill_name": skill.name,
                "instructions": skill.load_body()}
