"""意图路由器（第6期）：LLM 分类 → 子 Agent 标识。

工程细节：
- temperature=0 + max_tokens=10：分类任务要确定性，不需要自由度；
- 输出解析用"受控词表包含匹配 + 默认兜底"：模型多吐一个字也不会崩；
- 红旗症状在路由之前已被规则短路（见 base.handle_redflag），
  路由层的 emergency 分类是对规则表的 LLM 兜底（表达方式千变万化）。
"""

from __future__ import annotations

from typing import Optional

from app.llm.client import BaseLLMClient
from app.prompts.agents import ROUTER_PROMPT

VALID_AGENTS = ("appointment", "medication", "report", "emergency")
DEFAULT_AGENT = "appointment"


class Router:
    def __init__(self, client: BaseLLMClient):
        self.client = client

    def route(self, user_input: str, history: Optional[list[dict]] = None) -> str:
        """system 承载规则与最近对话，user 只放当前消息 ——
        消息角色分离让"指令"与"数据"互不污染（也是 mock 可测的前提）。"""
        system = ROUTER_PROMPT
        if history:
            recent = [m for m in history[-4:]
                      if m.get("role") in ("user", "assistant")
                      and m.get("content") and len(m.get("content", "")) < 200]
            if recent:
                system += "\n\n最近对话：\n" + "\n".join(
                    f"{'用户' if m['role'] == 'user' else '客服'}: {m['content']}"
                    for m in recent)
        resp = self.client.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user_input}],
            temperature=0.0, max_tokens=10, purpose="router")
        raw = (resp.content or "").strip().lower()
        for agent in VALID_AGENTS:
            if agent in raw:
                return agent
        return DEFAULT_AGENT
