"""LLM 事实提取（第7期）：短期/长期两条管线，均走 structured output。"""

from __future__ import annotations

from app.agent.memory.models import LongTermExtraction, ShortTermFacts
from app.agent.summarizer import render_transcript
from app.llm.client import BaseLLMClient, LLMError
from app.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT


def extract_short_term(client: BaseLLMClient, recent_messages: list[dict],
                       existing_facts: list[str]) -> list[str]:
    prompt = STM_EXTRACTION_PROMPT.format(
        existing_facts="；".join(existing_facts) or "（无）",
        transcript=render_transcript(recent_messages))
    try:
        parsed = client.parse_structured(
            [{"role": "user", "content": prompt}], ShortTermFacts,
            purpose="memory_stm")
        return [f for f in parsed.facts if isinstance(f, str) and f.strip()]
    except LLMError:
        return list(existing_facts)  # 提取失败：保持原状（降级而非清空）


def extract_long_term(client: BaseLLMClient, messages: list[dict],
                      summary, existing_ltm: list[str]):
    """返回 (facts: list[MemoryFactOut], interaction_summary: str)。"""
    prompt = LTM_EXTRACTION_PROMPT.format(
        existing_ltm="；".join(existing_ltm) or "（无）",
        summary=summary or "（无）",
        transcript=render_transcript(messages))
    try:
        parsed = client.parse_structured(
            [{"role": "user", "content": prompt}], LongTermExtraction,
            purpose="memory_ltm")
        return parsed.facts, parsed.interaction_summary
    except LLMError:
        return [], ""
