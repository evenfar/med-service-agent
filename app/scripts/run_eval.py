"""离线评估入口（第9期）。

用法：
  python -m app.scripts.run_eval                       # 离线规则模式（默认，零成本）
  python -m app.scripts.run_eval --mode multi          # Multi-Agent 模式
  python -m app.scripts.run_eval --judge               # 强制 LLM-judge（需真实 Key）
  python -m app.scripts.run_eval --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import Settings, is_offline  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 评估")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--judge", action="store_true",
                        help="强制启用 LLM-as-judge（需要真实 API Key）")
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    parser.set_defaults(judge=None)
    args = parser.parse_args()

    settings = Settings()
    offline = is_offline(settings) and not args.judge
    if args.judge and is_offline(settings):
        print("⚠️  请求 --judge 但未配置 API Key，仍将使用离线模式（judge 不可用）")

    dataset = args.dataset or settings.eval_dataset_path
    cases = load_dataset(dataset)
    sandbox = Sandbox(mode=args.mode, offline=offline)
    evaluator = Evaluator(sandbox, settings)
    report = evaluator.run_all(cases)

    print(f"\n══════ 评估报告（mode={args.mode}, judge={evaluator.use_judge}）══════")
    print(f"{'用例':<28}{'过程分':>8}{'结果分':>8}{'通过':>6}  说明")
    for cr in report.cases:
        proc = f"{cr.process_score:.2f}" if cr.process_score is not None else "-"
        res = f"{cr.result_score:.2f}" if cr.result_score is not None else "-"
        ok = "✅" if cr.passed else "❌"
        err = cr.error or ""
        print(f"{cr.case_id:<28}{proc:>8}{res:>8}{ok:>6}  {err}")
    print(f"\n合计: {report.passed}/{report.total} 通过 | "
          f"平均过程分 {report.avg_process or 0:.2f} | "
          f"平均结果分 {report.avg_result or 0:.2f}")
    if report.safety_failures:
        print(f"🚨 安全红线未达标用例: {', '.join(report.safety_failures)}")
    if report.errors:
        print(f"执行异常用例: {', '.join(report.errors)}")

    if args.output:
        payload = {
            "mode": args.mode, "total": report.total, "passed": report.passed,
            "avg_process": report.avg_process, "avg_result": report.avg_result,
            "safety_failures": report.safety_failures,
            "cases": [cr.__dict__ for cr in report.cases]}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {args.output}")


if __name__ == "__main__":
    main()
