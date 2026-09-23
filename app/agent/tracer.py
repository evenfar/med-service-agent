"""可观测：LLM/工具/路由事件记录 + token 计量。

对比 ecom 的改进：ecom 只在评估沙箱里用 monkey-patch 采集这些数据；
本项目把 tracer 作为一等公民注入 LLMClient 与 ToolRegistry，运行时就能看成本，
评估沙箱直接读 tracer（零补丁、零全局状态）—— 依赖注入带来的可测试性红利。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from typing import Any, Optional


class Tracer:
    def __init__(self, jsonl_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._t0 = time.time()
        self.jsonl_path = jsonl_path
        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

    # ---------- 记录 ----------

    def log(self, event_type: str, **payload: Any) -> None:
        rec = {"ts": round(time.time() - self._t0, 3), "type": event_type, **payload}
        with self._lock:
            self._entries.append(rec)
            if self.jsonl_path:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_llm(self, purpose: str, model: str, prompt_tokens: int,
                completion_tokens: int, latency_ms: float) -> None:
        self.log("llm_call", purpose=purpose, model=model,
                 prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                 latency_ms=round(latency_ms, 1))

    def log_tool(self, name: str, arguments: dict, ok: bool, latency_ms: float) -> None:
        self.log("tool_call", tool=name, arguments=arguments, ok=ok,
                 latency_ms=round(latency_ms, 1))

    def log_route(self, agent: str) -> None:
        self.log("route", agent=agent)

    # ---------- 查询 ----------

    @property
    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def since(self, index: int) -> list[dict]:
        """取 index 之后的事件（用于"本轮"范围内的分析，如引用校验）。"""
        with self._lock:
            return list(self._entries[index:])

    @property
    def cursor(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def llm_calls(self) -> list[dict]:
        return [e for e in self.entries if e["type"] == "llm_call"]

    @property
    def tool_calls(self) -> list[dict]:
        return [e for e in self.entries if e["type"] == "tool_call"]

    @property
    def total_tokens(self) -> int:
        return sum(e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
                   for e in self.llm_calls)

    def summary(self) -> str:
        usage: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0.0])
        for e in self.llm_calls:
            u = usage[e.get("purpose", "?")]
            u[0] += 1
            u[1] += e.get("prompt_tokens", 0)
            u[2] += e.get("completion_tokens", 0)
            u[3] += e.get("latency_ms", 0.0)
        lines = ["", "──── 运行摘要 ────",
                 f"{'用途':<10}{'调用':>6}{'prompt':>9}{'completion':>12}{'延迟ms':>10}"]
        for purpose, (n, pt, ct, lat) in sorted(usage.items()):
            lines.append(f"{purpose:<10}{n:>6}{pt:>9}{ct:>12}{lat:>10.0f}")
        total_n = sum(u[0] for u in usage.values())
        lines.append(f"LLM调用合计: {total_n} | 工具调用: {len(self.tool_calls)} 次 "
                     f"| 总token: {self.total_tokens}")
        return "\n".join(lines)
