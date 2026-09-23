"""知识库检索工具（第5期）：query → RAG 检索 → 带来源的结果。

安全要点：检索结果在进入模型上下文前先过 sanitize_tool_output（第10期），
包裹数据边界并剥离疑似注入内容 —— 工具输出是"数据"，不是"指令"。
"""

from __future__ import annotations

from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.safety import sanitize_tool_output


def search_knowledge(retriever: KnowledgeRetriever, query: str, top_k: int = 0) -> dict:
    if retriever is None:
        return {"success": False, "error": "知识库未启用"}
    k = top_k or 3
    try:
        chunks = retriever.search(query, top_k=k)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    results = [{"source": f"{c.chunk.doc}#{c.chunk.section}",
                "text": c.chunk.text,
                "sanitized": sanitize_tool_output(c.chunk.text)}
               for c in chunks]
    return {"success": True, "query": query, "chunks": results}
