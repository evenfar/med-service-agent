"""记忆提取的结构化 schema（第7期）。

对比 ecom 的改进：ecom 长期记忆提取让模型输出裸 JSON 再 json.loads，
失败即丢弃整次提取；本项目用 pydantic structured output —— 校验失败
由框架层的 fallback prompt 重试兜住，记忆提取成功率不再依赖模型自觉。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ShortTermFacts(BaseModel):
    """短期记忆提取结果。"""
    facts: list[str] = Field(default_factory=list,
                             description="本次对话新产生的关键事实，每条一句")


class MemoryFactOut(BaseModel):
    content: str = Field(description="事实内容，一句话")
    category: str = Field(description="分类，从 allergy/medication/chronic/history/preference/basic/other 中选")


class LongTermExtraction(BaseModel):
    """长期记忆巩固结果。"""
    facts: list[MemoryFactOut] = Field(default_factory=list)
    interaction_summary: str = Field(default="", description="本次会话一句话概括")
