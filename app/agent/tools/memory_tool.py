"""记忆查询工具（第7期）：recall_user_memory。

对比 ecom 的改进：ecom 用模块级 _memory_manager + set_memory_manager() 全局单例；
本项目通过构造注入（retriever/memory/skill 都在 build_tool_registry 时闭包捕获），
多实例并存、无隐藏全局状态。
"""

from __future__ import annotations

from typing import Optional

from app.agent.memory.manager import MemoryManager


def recall_user_memory(memory_manager: Optional[MemoryManager], query: str = "") -> dict:
    if memory_manager is None:
        return {"success": False, "error": "记忆功能未启用"}
    stm = [f for f in memory_manager.stm.facts]
    ltm = [{"content": f.content, "category": f.category}
           for f in memory_manager.ltm.facts]
    recent = memory_manager.ltm.interaction_summaries[-3:]
    return {"success": True, "query": query, "short_term": stm,
            "long_term": ltm, "recent_interactions": recent}
