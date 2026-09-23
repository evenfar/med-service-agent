"""文档切片（第5期）：Markdown → Chunk。

策略：按二级标题（##）切分，H3/H4 留在所属 H2 内保持语义完整
（检验指标类文档按"检查项目"自然分节）；单 chunk 超长时按空行贪心二次打包。
chunk_id 形如 {doc}#{idx:02d}-{sub:02d}，与内容解耦、稳定可增量更新。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200


@dataclass
class Chunk:
    chunk_id: str
    doc: str        # 文档名（不含扩展名）
    section: str    # 二级标题
    text: str

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc": self.doc,
                "section": self.section, "text": self.text}

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


def _split_by_h2(text: str) -> list[tuple[str, str]]:
    """返回 [(section, body), ...]；首个 H2 之前的内容作为「概览」。"""
    parts: list[tuple[str, str]] = []
    matches = list(re.finditer(r"^##\s+(.+)$", text, re.MULTILINE))
    if not matches:
        return [("概览", text.strip())] if text.strip() else []
    if matches[0].start() > 0:
        head = text[:matches[0].start()].strip()
        if head:
            parts.append(("概览", head))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((m.group(1).strip(), text[m.end():end].strip()))
    return parts


def _repack(section: str, body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body] if body else []
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    packs, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > max_chars:
            packs.append(current.strip())
            current = ""
        current += p + "\n\n"
    if current.strip():
        packs.append(current.strip())
    return packs


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    md_files = sorted(Path(kb_dir).glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"知识库目录没有任何 .md 文件: {kb_dir}")
    for md in md_files:
        doc = md.stem
        text = md.read_text(encoding="utf-8")
        for idx, (section, body) in enumerate(_split_by_h2(text)):
            for sub, pack in enumerate(_repack(section, body, MAX_CHUNK_CHARS)):
                chunk = Chunk(
                    chunk_id=f"{doc}#{idx:02d}-{sub:02d}",
                    doc=doc, section=section,
                    text=f"【{doc}#{section}】\n{pack}")
                chunks.append(chunk)
    return chunks
