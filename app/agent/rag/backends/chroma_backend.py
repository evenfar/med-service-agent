"""Chroma 向量数据库后端（第5期，可选安装）。

与 LocalBackend 同接口：教学要点在于"换后端不改一行业务代码"。
chromadb 为惰性导入 —— 未安装时给出明确指引而非 ImportError 裸抛。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.chunker import Chunk

_META_MODEL_KEY = "embedding_model"


class ChromaBackend(VectorBackend):
    def __init__(self, persist_dir: str | Path, collection: str):
        self._persist_dir = str(persist_dir)
        self._collection_name = collection
        self._client = None
        self._col = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb  # 惰性导入
        except ImportError as e:
            raise RuntimeError(
                "使用 RAG_BACKEND=chroma 需要先 pip install chromadb") from e
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               embedding_model: str) -> None:
        self._ensure_client()
        self._col.delete(where={"source": {"$ne": "__none__"}}) if self._col.count() else None
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"doc": c.doc, "section": c.section,
                        _META_MODEL_KEY: embedding_model} for c in chunks])

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        self._ensure_client()
        res = self._col.query(query_embeddings=[query_vector], n_results=top_k,
                              include=["documents", "metadatas", "distances"])
        out: list[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            chunk = Chunk(chunk_id=cid, doc=meta.get("doc", ""),
                          section=meta.get("section", ""), text=text)
            out.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

    def size(self) -> int:
        self._ensure_client()
        return self._col.count()

    def load(self) -> None:
        if self.size() == 0:
            raise FileNotFoundError(
                "Chroma 集合为空，请先运行 python -m app.scripts.build_kb_index "
                "--backend chroma 构建索引")

    def expected_embedding_model(self) -> str:
        self._ensure_client()
        if self._col.count() == 0:
            return ""
        got = self._col.get(limit=1, include=["metadatas"])
        metas = got.get("metadatas") or []
        return metas[0].get(_META_MODEL_KEY, "") if metas else ""
