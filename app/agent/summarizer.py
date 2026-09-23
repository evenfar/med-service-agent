"""历史摘要（第2期）：把老消息压缩成一条 summary，控制上下文膨胀。

渲染 transcript 时把 tool_calls / tool 结果也编进去（压缩不能丢工具事实）。
"""

from __future__ import annotations

from typing import Optional

from app.llm.client import BaseLLMClient
from app.prompts.summarizer import SUMMARY_PROMPT


def render_transcript(messages: list[dict], max_tool_chars: int = 200) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            lines.append(f"用户：{m.get('content', '')}")
        elif role == "assistant":
            content = m.get("content") or ""
            calls = m.get("tool_calls") or []
            call_text = " ".join(
                f"[调用工具 {c['function']['name']}({c['function']['arguments']})]"
                for c in calls)
            lines.append(f"助手：{content} {call_text}".strip())
        elif role == "tool":
            text = str(m.get("content", ""))[:max_tool_chars]
            lines.append(f"工具结果：{text}")
    return "\n".join(lines)


def summarize(client: BaseLLMClient, old_messages: list[dict],
              prev_summary: Optional[str]) -> str:
    prompt = SUMMARY_PROMPT.format(
        prev_summary=prev_summary or "（无）",
        transcript=render_transcript(old_messages),
    )
    resp = client.chat([{"role": "user", "content": prompt}],
                       temperature=0.3, purpose="summarize")
    return resp.content.strip()
