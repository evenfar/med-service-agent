"""运行轨迹载体（第9期）。

对比 ecom 的改进：ecom 靠沙箱 monkey-patch OpenAI client 采集轨迹；
本项目的 agent 全链路写 tracer，沙箱直接读 —— 采集机制是设计出来的，
不是补丁打出来的（依赖注入 → 可观测性 → 可评估性的递进关系）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.schemas.response import MedicalResponse


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class ToolObservation:
    name: str
    arguments: dict
    ok: bool
    result: str = ""


@dataclass
class RunTrace:
    case_id: str
    turns: list[str] = field(default_factory=list)
    final_response: Optional[MedicalResponse] = None
    route: Optional[str] = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.llm_calls)

    @property
    def num_llm_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def called_tools(self) -> list[str]:
        return [t.name for t in self.tool_observations]

    @property
    def succeeded(self) -> bool:
        return self.error == "" and self.final_response is not None

    @staticmethod
    def from_tracer(case_id: str, turns: list[str], tracer, final_response,
                    error: str = "") -> "RunTrace":
        trace = RunTrace(case_id=case_id, turns=turns,
                         final_response=final_response, error=error)
        for e in tracer.entries:
            if e["type"] == "llm_call":
                trace.llm_calls.append(LLMCallRecord(
                    purpose=e.get("purpose", "?"), model=e.get("model", ""),
                    prompt_tokens=e.get("prompt_tokens", 0),
                    completion_tokens=e.get("completion_tokens", 0),
                    latency_ms=e.get("latency_ms", 0.0)))
            elif e["type"] == "tool_call":
                trace.tool_observations.append(ToolObservation(
                    name=e.get("tool", "?"),
                    arguments=e.get("arguments", {}),
                    ok=bool(e.get("ok", False))))
            elif e["type"] == "route":
                trace.route = e.get("agent", "")
        return trace
