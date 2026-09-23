"""长期记忆（第7期）：跨会话健康档案，JSON per user 持久化。

写入校验（第10期知识点）：提取出的事实在落盘前过 validate_fact ——
① 内容长度受限；② 分类必须在受控枚举内；③ 携带指令式/导流样式的内容
直接丢弃（防"记忆投毒"：被污染的对话不该把恶意指令写成长期档案，
这正是 agent 记忆攻击研究的防御面，呼应 MedAgent-Memory-Attack 一类工作）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

from app.agent.safety.injection import looks_like_instruction

ALLOWED_CATEGORIES = {"allergy", "medication", "chronic", "history",
                      "preference", "basic", "other"}
MAX_FACT_CHARS = 100


@dataclass
class MemoryFact:
    content: str
    category: str
    created_at: str = ""
    source_session: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def validate_fact(content: str, category: str) -> tuple[bool, str]:
    """返回 (是否通过, 规范化后的category)。校验规则见模块 docstring。"""
    if not content or not content.strip():
        return False, category
    if len(content) > MAX_FACT_CHARS:
        return False, category
    if looks_like_instruction(content):
        return False, category
    return True, (category if category in ALLOWED_CATEGORIES else "other")


class LongTermMemory:
    def __init__(self, user_id: str = "default", memory_dir: str = "app/sessions/memory",
                 max_facts: int = 50):
        self.user_id = user_id
        self.memory_dir = memory_dir
        self.max_facts = max_facts
        self.facts: list[MemoryFact] = []
        self.interaction_summaries: list[dict] = []

    @property
    def path(self) -> str:
        return os.path.join(self.memory_dir, f"{self.user_id}.json")

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.facts = [MemoryFact(**d) for d in data.get("facts", [])]
            self.interaction_summaries = list(data.get("interaction_summaries", []))
        except (json.JSONDecodeError, OSError, TypeError):
            self.facts, self.interaction_summaries = [], []

    def save(self) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        payload = {"user_id": self.user_id,
                   "facts": [f.to_dict() for f in self.facts],
                   "interaction_summaries": self.interaction_summaries[-20:]}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add_facts(self, items: list[tuple[str, str]],
                  source_session: str = "") -> tuple[int, int]:
        """写入（带校验+去重+封顶）。返回 (写入条数, 被拒绝条数)。"""
        written = rejected = 0
        existing = {f.content.strip().lower() for f in self.facts}
        now = datetime.now().isoformat(timespec="seconds")
        for content, category in items:
            ok, norm_cat = validate_fact(content, category)
            if not ok:
                rejected += 1
                continue
            key = content.strip().lower()
            if key in existing:
                continue
            existing.add(key)
            self.facts.append(MemoryFact(content=content.strip(), category=norm_cat,
                                         created_at=now, source_session=source_session))
            written += 1
        if self.facts and len(self.facts) > self.max_facts:
            self.facts = self.facts[-self.max_facts:]  # 保留最新
        if written:
            self.save()
        return written, rejected

    def add_interaction_summary(self, summary: str) -> None:
        if not summary:
            return
        self.interaction_summaries.append(
            {"summary": summary[:200],
             "at": datetime.now().isoformat(timespec="seconds")})
        self.save()

    def build_prompt_section(self) -> str | None:
        if not self.facts:
            return None
        cat_label = {"allergy": "过敏史", "medication": "在用药品", "chronic": "慢病",
                     "history": "既往史", "preference": "偏好",
                     "basic": "基本情况", "other": "其他"}
        lines = [f"- [{cat_label.get(f.category, f.category)}] {f.content}"
                 for f in self.facts[-15:]]
        return ("## 用户健康档案（长期记忆，用药咨询前必须核对过敏史与在用药品）\n"
                + "\n".join(lines))

    def reset(self) -> None:
        self.facts = []
        self.interaction_summaries = []
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
