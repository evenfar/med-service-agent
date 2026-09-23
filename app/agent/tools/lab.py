"""检验报告工具（第3期）：查报告 + 危急值提示。"""

from __future__ import annotations

from app.agent.tools.mock_data import LAB_REPORTS


def query_lab_report(report_id: str) -> dict:
    rep = LAB_REPORTS.get(report_id)
    if not rep:
        return {"success": False,
                "error": f"未找到报告 {report_id}，请核实报告号（格式如 LAB-2026-001）"}
    has_critical = any(i["flag"] == "危急" for i in rep["items"])
    return {"success": True, "report": rep, "has_critical": has_critical}
