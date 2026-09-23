"""引用校验（第10期）：回答引用了知识库时，来源必须真实存在。

抗幻觉的结构化手段：报告/知识类回答里的【文档#小节】标记，
必须能在本轮实际检索到的 chunk 里找到；找不到的标记删除；
若本轮用了知识但回复没带任何来源，则补一个确定性的"参考来源"附录
（由代码生成，不是模型生成 —— 确定性工作握在自己手里）。
"""

from __future__ import annotations

import re

_CITE_RE = re.compile(r"【([^#】]+)#([^】]+)】")


def extract_citations(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _CITE_RE.finditer(text)]


def validate_citations(reply: str, used_sources: list[str]) -> tuple[str, list[str]]:
    """校验并修正引用。

    used_sources: 本轮检索到的真实来源列表，形如 ["就诊指南#挂号流程", ...]。
    返回 (修正后的回复, 被删除的伪造引用)。
    """
    valid = set(used_sources)
    fakes: list[str] = []

    def _sub(m: re.Match) -> str:
        src = f"{m.group(1)}#{m.group(2)}"
        if src in valid:
            return m.group(0)
        fakes.append(src)
        return ""

    fixed = _CITE_RE.sub(_sub, reply)
    return fixed, fakes


def append_sources_if_missing(reply: str, used_sources: list[str]) -> str:
    """用了知识库但没标注来源 → 代码生成参考来源附录。"""
    if not used_sources:
        return reply
    if extract_citations(reply):
        return reply
    lines = "\n".join(f"- 【{s}】" for s in dict.fromkeys(used_sources))
    return reply.rstrip() + "\n\n参考来源：\n" + lines
