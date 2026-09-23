"""知识检索器（第5期）：问句向量化 → 委托后端 → 一致性校验。

一致性校验是生产级细节：索引里记录了构建时的 embedding 模型名，
加载时与当前 embedder 比对，不一致直接报错 —— 避免"换了 embedding 但
还在用旧索引"这种检索质量悄悄劣化的隐蔽故障（沉默的错误最难排查）。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.backends.chroma_backend import ChromaBackend
from app.agent.rag.backends.local_backend import LocalBackend
from app.agent.rag.chunker import Chunk, chunk_markdown_dir
from app.agent.rag.embedder import BaseEmbedder, build_embedder
from app.agent.tracer import Tracer
from app.config.settings import Settings

__all__ = ["KnowledgeRetriever", "RetrievedChunk", "Chunk",
           "build_retriever", "build_index"]


class KnowledgeRetriever:
    def __init__(self, embedder: BaseEmbedder, backend: VectorBackend,
                 top_k: int = 3):
        self._embedder = embedder
        self._backend = backend
        self._top_k = top_k
        self._loaded = False

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def embedder_model(self) -> str:
        return self._embedder.model

    @property
    def size(self) -> int:
        return self._backend.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._backend.load()
        expected = self._backend.expected_embedding_model()
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引由 {expected} 构建，与当前 Embedder({self._embedder.model}) "
                f"不一致 —— 向量空间不同，检索结果不可信。请重建索引。")
        self._loaded = True

    def search(self, query: str, top_k: int = 0) -> list[RetrievedChunk]:
        self.load()
        q_vec = self._embedder.encode_one(query)
        return self._backend.search(q_vec, top_k or self._top_k)


def build_backend(settings: Settings) -> VectorBackend:
    if settings.rag_backend == "chroma":
        return ChromaBackend(settings.chroma_persist_dir, settings.chroma_collection)
    return LocalBackend(settings.kb_index_path)


def build_retriever(settings: Settings,
                    tracer: Tracer | None = None) -> KnowledgeRetriever:
    retriever = KnowledgeRetriever(build_embedder(settings, tracer),
                                   build_backend(settings), settings.rag_top_k)
    # 离线模式的体验兜底：本地索引缺失时用 HashEmbedder 零成本自动构建，
    # 保证"开箱即跑"；真实 embedding 模式不自动建（避免静默烧 API），提示走脚本。
    from app.config.settings import is_offline
    if is_offline(settings) and settings.rag_backend == "local" \
            and not Path(settings.kb_index_path).exists():
        build_index(settings, tracer)
    return retriever


def build_index(settings: Settings,
                tracer: Tracer | None = None) -> tuple[KnowledgeRetriever, int]:
    """离线建索引：chunk → embed → upsert。返回 (retriever, chunk数)。"""
    chunks = chunk_markdown_dir(settings.kb_dir)
    embedder = build_embedder(settings, tracer)
    vectors = embedder.encode([c.text for c in chunks])
    backend = build_backend(settings)
    backend.upsert(chunks, vectors, embedder.model)
    return KnowledgeRetriever(embedder, backend, settings.rag_top_k), len(chunks)
