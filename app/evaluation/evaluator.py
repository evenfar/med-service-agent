"""评估器（第9期）：跑用例 → 双层评分 → 聚合报告。

聚合规则：
- 过程分 = 非None项均值(tool_accuracy, tool_efficiency, process_soundness, route_match)
- 结果分 = 非None项均值(intent_match, keyword_coverage, requires_human_match,
  answer_quality, faithfulness, safety_guard)
- passed = 所有被指定维度均 ≥ threshold 且 token 在预算内。
  safety_guard 不达标时直接 fail（安全一票否决）。
离线模式自动跳过 judge（use_judge 且非离线才启用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import Settings, is_offline
from app.evaluation.dataset import EvalCase
from app.evaluation.metrics import (intent_match, judge_answer_quality,
                                    judge_faithfulness, judge_process_soundness,
                                    judge_safety, keyword_coverage,
                                    requires_human_match, route_match,
                                    safety_guard_pass, token_cost_pass,
                                    tool_accuracy, tool_efficiency)
from app.agent.tracer import Tracer
from app.evaluation.sandbox import Sandbox
from app.evaluation.trace import RunTrace
from app.llm.client import build_client


@dataclass
class CaseResult:
    case_id: str
    description: str = ""
    error: str = ""
    passed: bool = False
    process_score: Optional[float] = None
    result_score: Optional[float] = None
    detail: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    avg_process: Optional[float] = None
    avg_result: Optional[float] = None
    safety_failures: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)


class Evaluator:
    def __init__(self, sandbox: Sandbox, settings: Optional[Settings] = None):
        self.sandbox = sandbox
        self.settings = settings or Settings()
        offline = is_offline(self.settings)
        # 离线模式强制关闭 judge（MockLLM 不适合当裁判）
        self.use_judge = self.settings.eval_use_judge and not offline
        self._judge_client = (build_client(self.settings, Tracer())
                              if self.use_judge else None)

    # ---------- 单条 ----------

    def run_case(self, case: EvalCase) -> CaseResult:
        trace: RunTrace = self.sandbox.run(case)
        cr = CaseResult(case_id=case.id, description=case.description,
                        error=trace.error)
        if trace.error or trace.final_response is None:
            return cr
        resp = trace.final_response
        reply = resp.reply
        user_input = case.turns[-1] if case.turns else ""

        d: dict = {}
        d["tool_accuracy"] = tool_accuracy(case.expected_tools, trace.called_tools)
        d["tool_efficiency"] = tool_efficiency(case.min_tool_calls,
                                               len(trace.called_tools))
        d["token_pass"] = token_cost_pass(trace.total_tokens, case.max_tokens)
        d["route_match"] = route_match(case.expected_route, trace.route)
        d["intent_match"] = intent_match(case.expected_intent, resp.intent.value)
        d["keyword_coverage"] = keyword_coverage(case.expected_keywords, reply)
        d["requires_human_match"] = requires_human_match(
            case.expected_requires_human, resp.requires_human)
        d["safety_guard"] = safety_guard_pass(case, trace)
        if self.use_judge and self._judge_client is not None:
            quality, q_reason = judge_answer_quality(
                self._judge_client, user_input, reply, case.expected_keywords)
            faithful, f_reason = judge_faithfulness(
                self._judge_client, reply, trace.tool_observations)
            safe, s_reason = judge_safety(self._judge_client, reply)
            sound, p_reason = judge_process_soundness(
                self._judge_client, user_input, trace.called_tools)
            d.update({"answer_quality": quality, "faithfulness": faithful,
                      "judge_safety": safe, "process_soundness": sound})
            d["_reasons"] = {"quality": q_reason, "faithfulness": f_reason,
                             "safety": s_reason, "process": p_reason}

        process_keys = ["tool_accuracy", "tool_efficiency", "process_soundness",
                        "route_match"]
        result_keys = ["intent_match", "keyword_coverage", "requires_human_match",
                       "answer_quality", "faithfulness", "safety_guard"]
        cr.process_score = _mean([d[k] for k in process_keys if d.get(k) is not None])
        cr.result_score = _mean([d[k] for k in result_keys if d.get(k) is not None])
        threshold = self.settings.eval_pass_threshold
        scores = [v for k, v in d.items()
                  if isinstance(v, float) and not k.startswith("_")]
        cr.passed = bool(scores) and all(v >= threshold for v in scores) \
            and (d["token_pass"] is not False) \
            and (d["safety_guard"] != 0.0)
        cr.detail = {k: v for k, v in d.items()
                     if v is not None and not k.startswith("_")}
        return cr

    # ---------- 批量 ----------

    def run_all(self, cases: list[EvalCase]) -> EvalReport:
        report = EvalReport(total=len(cases))
        proc, res = [], []
        for case in cases:
            cr = self.run_case(case)
            report.cases.append(cr)
            if cr.error:
                report.errors.append(f"{case.id}: {cr.error}")
                continue
            report.passed += 1 if cr.passed else 0
            if cr.process_score is not None:
                proc.append(cr.process_score)
            if cr.result_score is not None:
                res.append(cr.result_score)
            if cr.detail.get("safety_guard") == 0.0:
                report.safety_failures.append(case.id)
        report.avg_process = _mean(proc)
        report.avg_result = _mean(res)
        return report


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None
