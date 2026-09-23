"""挂号/预约工具（第3期）：查询 + 取消/改期（写操作，会修改 mock 状态）。"""

from __future__ import annotations

from typing import Optional

from app.agent.tools.mock_data import APPOINTMENTS


def query_appointment(appointment_id: str) -> dict:
    apt = APPOINTMENTS.get(appointment_id)
    if not apt:
        return {"success": False,
                "error": f"未找到预约单 {appointment_id}，请核实单号（格式如 GH-2026-001）"}
    return {"success": True, "appointment": apt}


def cancel_appointment(appointment_id: str, action: str = "cancel",
                       new_date: Optional[str] = None) -> dict:
    """action: cancel=取消 / reschedule=改期。写操作示例：真实修改 mock 状态。"""
    apt = APPOINTMENTS.get(appointment_id)
    if not apt:
        return {"success": False, "error": f"未找到预约单 {appointment_id}"}
    if apt["status"] in ("cancelled", "completed"):
        return {"success": False,
                "error": f"该预约已是 {apt['status']} 状态，无法再操作"}
    if action == "cancel":
        apt["status"] = "cancelled"
        return {"success": True, "message": f"预约 {appointment_id}（{apt['department']} "
                f"{apt['doctor']} {apt['date']}）已取消，费用将在1-3个工作日原路退回"}
    if action == "reschedule":
        if not new_date:
            return {"success": False, "error": "改期必须提供 new_date（如 2026-10-08）"}
        apt["date"] = new_date
        return {"success": True, "message": f"预约 {appointment_id} 已改期至 {new_date}，"
                f"科室与医生保持不变（{apt['department']} {apt['doctor']}）"}
    return {"success": False, "error": f"不支持的操作: {action}（仅支持 cancel/reschedule）"}
