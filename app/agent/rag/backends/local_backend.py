"""本地向量后端（第5期）：JSON 持久化 + 全量余弦打分。

适用：教学（<100行全程透明）/ 小规模（<1k chunks 全量打分耗时可忽略）/ 零依赖。
不适用：万级规模（需 HNSW 近似检索）/ 多进程并发写（JSON 无锁）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.chunker import Chunk


class LocalBackend(VectorBackend):
    def __init__(self, index_path: str | Path):
        self._index_path = Path(index_path)
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._embedding_model: str = ""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               embedding_model: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 长度不一致: {len(chunks)} vs {len(vectors)}")
        self._chunks, self._vectors = list(chunks), list(vectors)
        self._embedding_model = embedding_model
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"embedding_model": embedding_model,
                   "chunks": [c.to_dict() for c in chunks],
                   "vectors": vectors}
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._index_path)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            self.load()
        scored = [(i, _cosine(query_vector, v)) for i, v in enumerate(self._vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievedChunk(chunk=self._chunks[i], score=s)
                for i, s in scored[:top_k]]

    def size(self) -> int:
        if not self._chunks and self._index_path.exists():
            self.load()
        return len(self._chunks)

    def load(self) -> None:
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"知识库索引不存在: {self._index_path}\n"
                "请先运行 python -m app.scripts.build_kb_index 构建索引")
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._embedding_model = data.get("embedding_model", "")
        self._chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        self._vectors = data["vectors"]

    def expected_embedding_model(self) -> str:
        if not self._embedding_model and self._index_path.exists():
            self.load()
        return self._embedding_model


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
