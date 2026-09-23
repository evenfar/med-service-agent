"""MCP Server（第4期）：FastMCP + Streamable HTTP，独立微服务。

只暴露只读工具（查询类）—— 写操作（取消预约）与有状态工具（记忆/技能）
不出本地进程，这是工具的权限边界设计：微服务化意味着任何人连上即可调用，
所以只外发"无副作用"的能力。
离线可跑：server 也走 settings 工厂，无 Key 时用 HashEmbedder+本地索引。
启动：python mcp_server/server.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from mcp.server.fastmcp import FastMCP  # mcp v1
except ModuleNotFoundError as e:  # mcp v2 改名 MCPServer —— 本项目按 v1 API 锁定
    raise ModuleNotFoundError(
        "本项目 MCP server 基于 mcp v1 API（FastMCP）。"
        "请安装 mcp>=1.8,<2（见 requirements.txt）") from e

from app.agent.rag.retriever import build_retriever  # noqa: E402
from app.agent.tools.appointment import query_appointment  # noqa: E402
from app.agent.tools.department import query_department  # noqa: E402
from app.agent.tools.lab import query_lab_report  # noqa: E402
from app.agent.tools.medicine import check_drug_interaction, query_medicine  # noqa: E402
from app.config.settings import Settings  # noqa: E402

mcp = FastMCP("med-tools", host="127.0.0.1", port=9301)
_settings = Settings()
_retriever = build_retriever(_settings)


@mcp.tool()
def query_appointment_tool(appointment_id: str) -> str:
    """查询挂号预约单详情（科室/医生/时间/状态）。单号格式 GH-2026-001。"""
    return json.dumps(query_appointment(appointment_id), ensure_ascii=False)


@mcp.tool()
def query_medicine_tool(name: str) -> str:
    """查询药品说明（适应症/用法/注意事项）。"""
    return json.dumps(query_medicine(name), ensure_ascii=False)


@mcp.tool()
def check_drug_interaction_tool(drug_a: str, drug_b: str) -> str:
    """核查两种药品联用风险等级与建议。"""
    return json.dumps(check_drug_interaction(drug_a, drug_b), ensure_ascii=False)


@mcp.tool()
def query_lab_report_tool(report_id: str) -> str:
    """查询检验报告（指标/参考范围/异常标记/危急值）。"""
    return json.dumps(query_lab_report(report_id), ensure_ascii=False)


@mcp.tool()
def query_department_tool(keyword: str) -> str:
    """按症状关键词推荐就诊科室。"""
    return json.dumps(query_department(keyword), ensure_ascii=False)


@mcp.tool()
def search_knowledge_tool(query: str) -> str:
    """检索医院知识库（就诊/用药/指标/科室），返回带来源片段。"""
    try:
        chunks = _retriever.search(query, top_k=3)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)
    results = [{"source": f"{c.chunk.doc}#{c.chunk.section}", "text": c.chunk.text}
               for c in chunks]
    return json.dumps({"success": True, "query": query, "chunks": results},
                      ensure_ascii=False)


if __name__ == "__main__":
    print(f"MCP server 启动: http://127.0.0.1:9301/mcp "
          f"(offline={not _settings.openai_api_key})")
    mcp.run(transport="streamable-http")
