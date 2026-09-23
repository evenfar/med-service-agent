"""pytest 共享夹具：全部离线（MockLLM + HashEmbedder），零 API 依赖。

session 级只建一次知识库索引；用例级拿到互不串扰的临时 session/memory 路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 相对路径(知识库/索引)以项目根为基准

from app.agent.rag.retriever import build_index  # noqa: E402
from app.config.settings import Settings  # noqa: E402


@pytest.fixture(scope="session")
def shared_index(tmp_path_factory) -> str:
    """全 session 共享的知识库索引（只读）。"""
    d = tmp_path_factory.mktemp("kb_index")
    s = Settings(mock_mode=True, kb_index_path=str(d / "kb_index.json"))
    _, n = build_index(s)
    assert n > 0
    return s.kb_index_path


@pytest.fixture
def agent_settings(tmp_path, shared_index) -> Settings:
    """每用例独立的离线 Settings（隔离 session 与 memory 目录）。"""
    return Settings(
        mock_mode=True,
        kb_index_path=shared_index,
        session_path=str(tmp_path / "session.json"),
        memory_dir=str(tmp_path / "memory"),
    )
