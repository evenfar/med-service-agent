"""短期记忆（第7期）：会话内事实列表，随会话文件持久化。"""

from __future__ import annotations

from app.agent.memory.extraction import extract_short_term
from app.llm.client import BaseLLMClient


class ShortTermMemory:
    def __init__(self, facts: list[str] | None = None):
        self.facts: list[str] = list(facts or [])

    def update(self, client: BaseLLMClient, recent_messages: list[dict]) -> None:
        """每轮对话后：LLM 从最近消息提取事实并与已有合并去重。"""
        new_facts = extract_short_term(client, recent_messages, self.facts)
        merged = list(self.facts)
        for f in new_facts:
            if f not in merged:
                merged.append(f)
        self.facts = merged[-20:]  # 短期记忆封顶，防爆长

    def build_prompt_section(self) -> str | None:
        if not self.facts:
            return None
        lines = "\n".join(f"- {f}" for f in self.facts[-10:])
        return "## 本会话已知事实（短期记忆）\n" + lines

    def reset(self) -> None:
        self.facts = []

    def to_dict(self) -> dict:
        return {"facts": self.facts}

    @staticmethod
    def from_dict(data: dict) -> "ShortTermMemory":
        return ShortTermMemory((data or {}).get("facts", []))
