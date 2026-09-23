"""向量后端抽象（第5期）：storage 与打分策略解耦。

两个实现：LocalBackend（手写余弦+JSON，零依赖默认）/ ChromaBackend（向量库）。
retriever 只依赖本接口 —— 这就是"可切换后端"的全部秘密：一个抽象类 + 工厂。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.agent.rag.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorBackend(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               embedding_model: str) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def expected_embedding_model(self) -> str: ...


def backend_path_default() -> Path:
    raise NotImplementedError  # 由 settings 注入具体路径
