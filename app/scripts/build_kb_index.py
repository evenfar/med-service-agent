"""离线构建知识库索引（第5期）。

用法：
  python -m app.scripts.build_kb_index              # 默认后端(local)
  python -m app.scripts.build_kb_index --backend chroma
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.retriever import build_index  # noqa: E402
from app.agent.tracer import Tracer  # noqa: E402
from app.config.settings import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="构建知识库向量索引")
    parser.add_argument("--backend", choices=["local", "chroma"], default=None)
    args = parser.parse_args()

    settings = Settings()
    if args.backend:
        settings.rag_backend = args.backend
    retriever, n = build_index(settings, Tracer())
    print(f"✅ 索引构建完成: {n} 个 chunk，后端={settings.rag_backend}，"
          f"embedding={retriever.embedder_model}")
    print(f"   验证: 检索「医保怎么报销」→ "
          f"{[f'{c.chunk.doc}#{c.chunk.section}' for c in retriever.search('医保怎么报销', top_k=2)]}")


if __name__ == "__main__":
    main()
