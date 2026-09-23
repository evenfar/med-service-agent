"""RAG 测试（第5期）：切片/向量化/后端/检索器一致性校验。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.rag.backends.local_backend import LocalBackend, _cosine
from app.agent.rag.chunker import Chunk, chunk_markdown_dir
from app.agent.rag.embedder import HashEmbedder
from app.agent.rag.retriever import KnowledgeRetriever, build_index
from app.config.settings import Settings

KB_DIR = Path("app/agent/rag/knowledge")


class TestChunker:
    def test_chunks_have_doc_and_section(self):
        chunks = chunk_markdown_dir(KB_DIR)
        assert len(chunks) > 10
        docs = {c.doc for c in chunks}
        assert docs == {"就诊指南", "用药安全", "检验指标解读", "科室导航"}
        for c in chunks:
            assert c.chunk_id and c.section
            assert c.text.startswith(f"【{c.doc}#{c.section}】")

    def test_chunk_size_capped(self):
        for c in chunk_markdown_dir(KB_DIR):
            assert len(c.text) < 1400  # MAX_CHUNK_CHARS + 标签余量


class TestHashEmbedder:
    def test_deterministic_and_normalized(self):
        e = HashEmbedder(32)
        v1 = e.encode_one("医保报销流程")
        v2 = e.encode_one("医保报销流程")
        assert v1 == v2  # 同一进程内一致
        assert abs(sum(x * x for x in v1) - 1.0) < 1e-6  # 单位向量
        assert len(v1) == 32

    def test_different_texts_differ(self):
        e = HashEmbedder(64)
        assert e.encode_one("医保") != e.encode_one("骨科手术")

    def test_cross_process_stability(self):
        """MD5 散列保证索引跨进程可复用（内置 hash 做不到）。"""
        import subprocess, sys
        code = ("from app.agent.rag.embedder import HashEmbedder;"
                "print(HashEmbedder(16).encode_one('医保报销')[0])")
        a = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=".")
        b = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=".")
        assert a.stdout == b.stdout and a.stdout.strip()


class TestLocalBackend:
    def test_upsert_search_roundtrip(self, tmp_path):
        backend = LocalBackend(tmp_path / "idx.json")
        chunks = [Chunk("a#00-00", "docA", "S1", "【docA#S1】内容一"),
                  Chunk("b#00-00", "docB", "S2", "【docB#S2】内容二")]
        vecs = [[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6]
        backend.upsert(chunks, vecs, "hash-8")

        reloaded = LocalBackend(tmp_path / "idx.json")  # 从磁盘恢复
        hits = reloaded.search([1.0] + [0.0] * 7, top_k=2)
        assert hits[0].chunk.doc == "docA" and hits[1].chunk.doc == "docB"
        assert reloaded.expected_embedding_model() == "hash-8"

    def test_missing_index_raises_with_hint(self, tmp_path):
        backend = LocalBackend(tmp_path / "none.json")
        with pytest.raises(FileNotFoundError, match="build_kb_index"):
            backend.search([0.0], 3)

    def test_cosine(self):
        assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
        assert _cosine([0, 0], [1, 1]) == 0.0  # 零向量防除零


class TestRetrieverConsistency:
    def test_model_mismatch_rejected(self, tmp_path):
        backend = LocalBackend(tmp_path / "idx.json")
        backend.upsert([Chunk("a#00-00", "a", "s", "t")],
                       [[0.1, 0.2]], "text-embedding-3-small")
        retriever = KnowledgeRetriever(HashEmbedder(), backend)
        with pytest.raises(ValueError, match="不一致"):
            retriever.search("任意")

    def test_build_index_and_retrieve(self, tmp_path):
        s = Settings(mock_mode=True, kb_index_path=str(tmp_path / "kb.json"))
        retriever, n = build_index(s)
        assert n >= 12
        hits = retriever.search("医保怎么报销", top_k=2)
        assert hits and hits[0].chunk.doc == "就诊指南"
        src = f"{hits[0].chunk.doc}#{hits[0].chunk.section}"
        assert src == "就诊指南#医保报销"
