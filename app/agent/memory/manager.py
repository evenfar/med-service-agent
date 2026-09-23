"""MemoryManager（第7期）：短期+长期记忆的统一门面。"""

from __future__ import annotations

from app.agent.memory.extraction import extract_long_term
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.short_term import ShortTermMemory
from app.llm.client import BaseLLMClient


class MemoryManager:
    def __init__(self, client: BaseLLMClient, user_id: str = "default",
                 memory_dir: str = "app/sessions/memory",
                 enabled: bool = True, max_ltm_facts: int = 50):
        self.client = client
        self.enabled = enabled
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory(user_id, memory_dir, max_ltm_facts)
        if enabled:
            self.ltm.load()

    def update_short_term(self, recent_messages: list[dict]) -> None:
        if self.enabled:
            self.stm.update(self.client, recent_messages)

    def consolidate_to_long_term(self, messages: list[dict], summary) -> None:
        """会话结束时：LLM 提取 → 校验写入 → 记录交互摘要。"""
        if not self.enabled:
            return
        facts, interaction_summary = extract_long_term(
            self.client, messages, summary,
            [f.content for f in self.ltm.facts])
        items = [(f.content, f.category) for f in facts]
        written, rejected = self.ltm.add_facts(items)
        self.ltm.add_interaction_summary(interaction_summary)
        if written or rejected:
            print(f"🧠 长期记忆：写入{written}条，拦截{rejected}条可疑内容")

    def build_prompt_sections(self) -> list[dict]:
        if not self.enabled:
            return []
        sections = []
        ltm = self.ltm.build_prompt_section()
        if ltm:
            sections.append({"role": "system", "content": ltm})
        stm = self.stm.build_prompt_section()
        if stm:
            sections.append({"role": "system", "content": stm})
        return sections

    def reset_short_term(self) -> None:
        self.stm.reset()

    def stm_to_dict(self) -> dict:
        return self.stm.to_dict()

    def restore_stm(self, data: dict) -> None:
        self.stm = ShortTermMemory.from_dict(data)
