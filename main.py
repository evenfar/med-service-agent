"""CLI 入口：python main.py [--mock] [--multi]

命令：quit/exit 退出 | reset 重置会话 | skills 查看技能 | memory 查看记忆
"""

from __future__ import annotations

import sys

from app.config.settings import Settings
from app.schemas.response import INTENT_LABELS, IntentType, UrgencyLevel

URGENCY_LABELS = {"routine": "常规", "attention": "留意", "urgent": "紧急",
                  "emergency": "急症"}


def main() -> None:
    args = sys.argv[1:]
    settings = Settings()
    if "--mock" in args:
        settings.mock_mode = True
    if "--multi" in args:
        settings.multi_agent_enabled = True

    if settings.multi_agent_enabled:
        from app.multi_agent.orchestrator import MultiAgentOrchestrator
        agent = MultiAgentOrchestrator(settings)
        mode = "Multi-Agent（挂号/用药/报告/急诊 分流）"
    else:
        from app.agent.chat import MedicalAgent
        agent = MedicalAgent(settings)
        mode = "单 Agent"

    offline = settings.mock_mode or not settings.openai_api_key
    print("=" * 56)
    print(f"  互联网医院 · 健康助手「小医」({mode})")
    print(f"  模式: {'离线 mock' if offline else settings.model_name} | "
          f"工具/知识库/记忆/技能 已装配")
    print("  试试: 查预约 GH-2026-001 / 布洛芬怎么吃 / 华法林和阿司匹林能一起吃吗")
    print("        帮我看看报告 LAB-2026-004 / 医保报销流程 / 我突然胸痛")
    print("  命令: quit退出 · reset重置 · skills技能 · memory记忆")
    print("=" * 56)
    if agent.history_size:
        print(f"💬 已恢复上次会话（{agent.history_size} 条历史）")

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in ("quit", "exit"):
            break
        if low == "reset":
            agent.reset_session()
            print("会话已重置。")
            continue
        if low == "skills":
            sm = getattr(agent, "skill_manager", None)
            if sm and sm.enabled:
                print(f"已加载 {len(sm.skill_names)} 个技能: {sm.skill_names}")
            else:
                print("技能系统未启用")
            continue
        if low == "memory":
            mm = getattr(agent, "memory_manager", None)
            if mm and mm.enabled:
                print(f"短期记忆: {mm.stm.facts or '（无）'}")
                print(f"长期档案: {[f'[{f.category}]{f.content}' for f in mm.ltm.facts] or '（无）'}")
            else:
                print("记忆功能未启用")
            continue
        if low == "summary":
            print(f"历史摘要: {agent.summary or '（无，未触发压缩）'}")
            continue

        try:
            resp = agent.chat(user_input)
            print(f"\n🩺 小医: {resp.reply}")
            print(f"   [意图: {INTENT_LABELS.get(resp.intent, resp.intent.value)}"
                  f" | 置信度: {resp.confidence:.0%}"
                  f" | 紧急度: {URGENCY_LABELS.get(resp.urgency.value, resp.urgency.value)}"
                  f" | 转人工: {'是' if resp.requires_human else '否'}]")
            if resp.follow_up_question:
                print(f"   [追问: {resp.follow_up_question}]")
        except Exception as e:  # noqa: BLE001 —— 单轮失败不清空会话
            print(f"\n⚠️  出错了: {e}")

    agent.save()
    agent.close()
    print(agent.tracer.summary())
    print("再见，祝您健康！")


if __name__ == "__main__":
    main()
