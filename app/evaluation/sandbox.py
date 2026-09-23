"""评估沙箱（第9期）：隔离环境 + 轨迹采集。

隔离三件事（对比 ecom，它靠修改全局 settings 实现、且不还原）：
1. 全新 Settings 实例（memory/mcp 关闭，mock 可强制）——不碰任何全局状态；
2. 每条用例独立临时 session 文件；
3. 内存 Tracer（不落盘）。
关键决策：跑完不调用 agent.close() —— 那会触发长期记忆巩固（一次 LLM 写入，
污染且花钱）。采集直接读 tracer，无需任何 monkey-patch。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from app.config.settings import Settings
from app.evaluation.dataset import EvalCase
from app.evaluation.trace import RunTrace


class Sandbox:
    def __init__(self, mode: str = "single", tmp_root: Optional[str] = None,
                 offline: bool = True, kb_index_path: Optional[str] = None):
        self.mode = mode  # single / multi
        self.offline = offline
        self.kb_index_path = kb_index_path
        self.tmp_root = Path(tmp_root) if tmp_root else Path(
            tempfile.mkdtemp(prefix="med_eval_"))
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def _settings(self, case: EvalCase) -> Settings:
        kwargs = dict(memory_enabled=False, mcp_enabled=False,
                      multi_agent_enabled=(self.mode == "multi"))
        if self.offline:
            kwargs["mock_mode"] = True
        if self.kb_index_path:
            kwargs["kb_index_path"] = self.kb_index_path
        return Settings(**kwargs)

    def run(self, case: EvalCase) -> RunTrace:
        from app.agent.tools.mock_data import reset_mock_data
        from app.agent.tracer import Tracer
        reset_mock_data()  # 每条用例重置模拟数据，保证可复现（写操作不留状态）
        settings = self._settings(case)
        session_path = str(self.tmp_root / f"{case.id}.json")
        tracer = Tracer()
        agent = None
        try:
            if self.mode == "multi":
                from app.multi_agent.orchestrator import MultiAgentOrchestrator
                agent = MultiAgentOrchestrator(settings, session_path=session_path,
                                               tracer=tracer)
            else:
                from app.agent.chat import MedicalAgent
                agent = MedicalAgent(settings, session_path=session_path,
                                     tracer=tracer)
            result = None
            for turn in case.turns:
                result = agent.chat(turn)
            return RunTrace.from_tracer(case.id, case.turns, tracer, result)
        except Exception as e:  # noqa: BLE001 —— 单条用例失败不中断整轮评估
            if agent is not None:
                return RunTrace.from_tracer(case.id, case.turns, tracer, None,
                                            error=f"{type(e).__name__}: {e}")
            return RunTrace(case_id=case.id, turns=case.turns,
                            error=f"{type(e).__name__}: {e}")
