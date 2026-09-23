"""工具注册表（第3期）：OpenAI function calling 协议 + 统一执行入口。

设计要点：
1. 每个工具 = (name, description, JSON Schema parameters, fn) 一条记录，
   definitions 属性直接产出 OpenAI tools 格式 —— 单一事实来源；
2. execute() 是唯一入口：必填/类型校验 → 执行 → 一切异常包装成 {"error": ...}
   JSON 观察结果（工具崩溃不升级为 agent 崩溃，模型还有机会自我纠正）；
3. subset(names) 支持子 Agent 工具白名单隔离（第6期）；
4. 全部调用进 tracer（延迟/成败），可观测是一等公民。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.agent.tracer import Tracer
from app.agent.tools.appointment import cancel_appointment, query_appointment
from app.agent.tools.department import query_department
from app.agent.tools.knowledge import search_knowledge
from app.agent.tools.lab import query_lab_report
from app.agent.tools.medicine import check_drug_interaction, query_medicine
from app.agent.tools.memory_tool import recall_user_memory
from app.agent.tools.skill_tool import load_skill

_TYPE_CHECK: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    @property
    def openai_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self, tracer: Optional[Tracer] = None):
        self._tools: dict[str, ToolDef] = {}
        self._tracer = tracer

    def register(self, name: str, description: str, parameters: dict,
                 fn: Callable[..., dict]) -> None:
        self._tools[name] = ToolDef(name, description, parameters, fn)

    @property
    def definitions(self) -> list[dict]:
        return [t.openai_schema for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def subset(self, allowed: set[str]) -> "ToolRegistry":
        """白名单视图：共享同一批 ToolDef，新的注册器只暴露被允许的子集。"""
        view = ToolRegistry(self._tracer)
        view._tools = {k: v for k, v in self._tools.items() if k in allowed}
        return view

    def execute(self, name: str, arguments: dict) -> str:
        """统一执行入口，永不抛异常；返回 JSON 字符串观察结果。"""
        t0 = time.perf_counter()
        arguments = arguments if isinstance(arguments, dict) else {}
        tool = self._tools.get(name)
        if tool is None:
            return self._finish(name, arguments, False, t0,
                                {"error": f"未知工具: {name}，可用: {self.names()}"})
        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        for param in required:
            if param not in arguments:
                return self._finish(name, arguments, False, t0,
                                    {"error": f"缺少必填参数: {param}"})
        for param, value in arguments.items():
            spec = props.get(param)
            if spec and not _TYPE_CHECK.get(spec.get("type", "string"),
                                            lambda v: True)(value):
                return self._finish(name, arguments, False, t0,
                                    {"error": f"参数 {param} 类型应为 {spec.get('type')}"})
        filtered = {k: v for k, v in arguments.items() if k in props}
        try:
            result = tool.fn(**filtered)
            ok = not (isinstance(result, dict) and result.get("success") is False)
            return self._finish(name, arguments, ok, t0, result)
        except Exception as e:  # noqa: BLE001 —— 工具异常必须降级为观察结果
            return self._finish(name, arguments, False, t0,
                                {"error": f"{type(e).__name__}: {e}"})

    def _finish(self, name: str, arguments: dict, ok: bool,
                t0: float, result: dict) -> str:
        if self._tracer:
            self._tracer.log_tool(name, arguments, ok,
                                  (time.perf_counter() - t0) * 1000)
        return json.dumps(result, ensure_ascii=False)


# ---------------- 工厂：装配 9 个工具（依赖全部注入，无全局单例） ----------------

@dataclass
class ToolDeps:
    """工具层的协作对象。None 表示该能力未启用，对应工具返回明确错误。"""
    tracer: Optional[Tracer] = None
    retriever: Any = None            # KnowledgeRetriever（第5期）
    memory_manager: Any = None       # MemoryManager（第7期）
    skill_manager: Any = None        # SkillManager（第8期）


def build_tool_registry(deps: ToolDeps) -> ToolRegistry:
    reg = ToolRegistry(deps.tracer)

    reg.register(
        "query_appointment",
        "查询挂号预约单的详情（科室/医生/时间/状态）。单号格式 GH-2026-001。",
        {"type": "object",
         "properties": {"appointment_id": {"type": "string", "description": "预约单号"}},
         "required": ["appointment_id"]},
        query_appointment)

    reg.register(
        "cancel_appointment",
        "取消或改期预约。action=cancel 取消并退费；action=reschedule 需给 new_date。",
        {"type": "object",
         "properties": {
             "appointment_id": {"type": "string"},
             "action": {"type": "string", "enum": ["cancel", "reschedule"],
                        "description": "cancel 或 reschedule"},
             "new_date": {"type": "string", "description": "改期目标日期，如 2026-10-08"}},
         "required": ["appointment_id", "action"]},
        cancel_appointment)

    reg.register(
        "query_medicine",
        "查询药品目录中某药品的说明（适应症/用法/注意事项）。仅收录常见教学药品。",
        {"type": "object",
         "properties": {"name": {"type": "string", "description": "药品名，如 布洛芬"}},
         "required": ["name"]},
        query_medicine)

    reg.register(
        "check_drug_interaction",
        "核查两种药品联用风险等级与建议。注意：返回「未收录」不等于无风险。",
        {"type": "object",
         "properties": {"drug_a": {"type": "string"}, "drug_b": {"type": "string"}},
         "required": ["drug_a", "drug_b"]},
        check_drug_interaction)

    reg.register(
        "query_lab_report",
        "查询检验报告（指标值/参考范围/异常标记）。报告号格式 LAB-2026-001。",
        {"type": "object",
         "properties": {"report_id": {"type": "string"}},
         "required": ["report_id"]},
        query_lab_report)

    reg.register(
        "query_department",
        "按症状或关键词推荐就诊科室（导诊建议，非诊断）。",
        {"type": "object",
         "properties": {"keyword": {"type": "string", "description": "症状或关键词"}},
         "required": ["keyword"]},
        query_department)

    reg.register(
        "search_knowledge",
        "检索医院知识库（就诊流程/医保政策/用药安全/指标解读），返回带来源的片段。"
        "回答政策与流程问题前必须先调用。",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "top_k": {"type": "integer", "description": "返回条数，默认3"}},
         "required": ["query"]},
        lambda query, top_k=0: search_knowledge(deps.retriever, query, top_k))

    reg.register(
        "recall_user_memory",
        "查询当前用户的健康档案（短期/长期记忆：过敏史、用药史等）。用药咨询前先调用。",
        {"type": "object",
         "properties": {"query": {"type": "string"}},
         "required": []},
        lambda query="": recall_user_memory(deps.memory_manager, query))

    reg.register(
        "load_skill",
        "按名称加载一项技能的完整操作流程（如 triage 分诊流程）。"
        "用户问题匹配技能场景时先调用本工具。",
        {"type": "object",
         "properties": {"skill_name": {"type": "string"}},
         "required": ["skill_name"]},
        lambda skill_name: load_skill(deps.skill_manager, skill_name))

    return reg
