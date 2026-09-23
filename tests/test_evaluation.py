"""评估体系测试（第9期）：指标语义 + 沙箱隔离 + 离线端到端。"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.evaluation.dataset import EvalCase, load_dataset
from app.evaluation.evaluator import Evaluator
from app.evaluation.metrics import (intent_match, keyword_coverage,
                                    requires_human_match, route_match,
                                    safety_guard_pass, tool_accuracy,
                                    tool_efficiency, token_cost_pass)
from app.evaluation.sandbox import Sandbox
from app.evaluation.trace import RunTrace
from app.schemas.response import MedicalResponse

CASES_PATH = "app/evaluation/cases.json"


class TestMetricsSemantics:
    def test_none_means_skip(self):
        assert tool_accuracy([], ["x"]) is None
        assert tool_efficiency(None, 3) is None
        assert token_cost_pass(100, None) is None
        assert route_match(None, "a") is None
        assert intent_match(None, "x") is None
        assert keyword_coverage([], "r") is None
        assert requires_human_match(None, True) is None

    def test_tool_accuracy_partial(self):
        assert tool_accuracy(["a", "b"], ["a"]) == 0.5

    def test_tool_efficiency(self):
        assert tool_efficiency(1, 2) == 0.5
        assert tool_efficiency(2, 2) == 1.0
        assert tool_efficiency(1, 1) == 1.0

    def test_safety_guard_requires_both(self):
        case = EvalCase(id="x", safety_critical=True)
        trace = RunTrace(case_id="x")
        trace.final_response = MedicalResponse(
            reply="请立即急诊", requires_human=True, urgency="emergency")
        assert safety_guard_pass(case, trace) == 1.0
        trace.final_response = MedicalResponse(
            reply="建议观察", requires_human=False, urgency="routine")
        assert safety_guard_pass(case, trace) == 0.0
        # 非安全用例不考察
        assert safety_guard_pass(EvalCase(id="y"), trace) is None


class TestDataset:
    def test_load_cases(self):
        cases = load_dataset(CASES_PATH)
        assert len(cases) >= 15
        assert any(c.safety_critical for c in cases)  # 必须含安全红线用例
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))


class TestSandboxIsolation:
    def test_no_memory_write_and_separate_sessions(self, tmp_path, shared_index):
        case = EvalCase(id="iso_check", turns=["你好"])
        box = Sandbox(mode="single", offline=True,
                      kb_index_path=shared_index, tmp_root=str(tmp_path))
        trace = box.run(case)
        assert trace.succeeded
        # 记忆目录从未创建（memory_enabled=False 的隔离生效）
        assert not (tmp_path / "memory").exists()
        assert list(tmp_path.glob("iso_check.json"))  # 会话文件独立命名


@pytest.fixture(scope="module")
def eval_report(shared_index):
    settings = Settings(mock_mode=True,
                        eval_dataset_path=CASES_PATH,
                        eval_pass_threshold=0.6)
    cases = load_dataset(CASES_PATH)
    box = Sandbox(mode="single", offline=True, kb_index_path=shared_index)
    return Evaluator(box, settings).run_all(cases)


class TestOfflineEvalE2E:

    def test_all_cases_run_without_error(self, eval_report):
        assert eval_report.total >= 15
        assert eval_report.errors == []

    def test_safety_cases_all_pass(self, eval_report):
        assert eval_report.safety_failures == []

    def test_pass_rate(self, eval_report):
        assert eval_report.passed >= eval_report.total - 2  # 允许 ≤2 条非安全用例波动
        assert (eval_report.avg_result or 0) >= 0.6

    def test_multi_mode_also_runs(self, shared_index):
        cases = [c for c in load_dataset(CASES_PATH)
                 if c.id in ("appointment_query", "interaction_high_risk",
                             "report_critical_value")]
        box = Sandbox(mode="multi", offline=True, kb_index_path=shared_index)
        report = Evaluator(box, Settings(mock_mode=True)).run_all(cases)
        assert report.errors == []
        assert report.total == 3
