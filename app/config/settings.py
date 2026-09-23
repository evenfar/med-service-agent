"""全局配置：pydantic-settings 从 .env / 环境变量读取。

工程要点（对比 ecom 的改进）：
- Settings 是"可多实例"的类：评估沙箱用 Settings(memory_enabled=False, mock_mode=True)
  构造独立配置，不修改全局单例 → 无状态泄漏（ecom 的沙箱直接改全局 settings 且不还原）。
- mock_mode 显式化：MOCK_MODE=1 或未配置 API Key 时自动进入离线模式，全链路零成本可跑。
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- LLM（OpenAI 兼容接口；默认智谱 GLM，可换 DeepSeek/OpenAI/vLLM） ----
    openai_api_key: str = ""
    openai_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model_name: str = "glm-4-flash"
    embedding_model: str = "embedding-2"  # 智谱；OpenAI 用 text-embedding-3-small
    temperature: float = 0.5

    # 离线 mock 模式：True 时 LLMClient/Embedder 全部走本地假实现，不需要 Key
    mock_mode: bool = False

    # ---- ReAct 循环 ----
    max_react_steps: int = 6

    # ---- LLM 重试 ----
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0  # 指数退避基数：1s, 2s, 4s

    # ---- MCP ----
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9301/mcp"

    # ---- RAG ----
    rag_backend: str = "local"  # local(手写余弦+JSON) / chroma(可选安装)
    kb_dir: str = "app/agent/rag/knowledge"
    kb_index_path: str = "app/sessions/kb_index.json"
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "med_kb"
    rag_top_k: int = 3

    # ---- Multi-Agent ----
    multi_agent_enabled: bool = False

    # ---- Memory ----
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50

    # ---- Skills ----
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # ---- 评估 ----
    eval_dataset_path: str = "app/evaluation/cases.json"
    eval_use_judge: bool = True
    eval_pass_threshold: float = 0.6

    # ---- 会话管理 ----
    session_path: str = "app/sessions/session.json"
    # 简化策略：按消息条数触发压缩（生产应按 token 预算，见 docs 第2期）
    history_threshold: int = 12
    history_keep_recent: int = 4

    # ---- 可观测 ----
    trace_path: str = "app/sessions/trace.jsonl"

    model_config = {"env_file": ".env", "extra": "ignore"}


def is_offline(settings: Settings) -> bool:
    """离线判定：显式 mock 或没有 Key 都进入离线模式（开箱即跑）。"""
    return settings.mock_mode or not settings.openai_api_key
