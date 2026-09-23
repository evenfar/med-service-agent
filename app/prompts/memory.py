"""记忆提取 Prompt（第7期）。输出由 structured output schema 约束。"""

STM_EXTRACTION_PROMPT = """从下面的对话中提取"本会话内值得记住的关键事实"：
用户提到的症状与持续时间、查过的单号（预约/报告）、正在咨询的药品、明确的偏好。
只提取对话中明确出现的信息，不要推测。已有事实：{existing_facts}
已存在的事实不要重复提取；没有新事实就返回空列表。

对话记录：
{transcript}"""

LTM_EXTRACTION_PROMPT = """从下面的完整会话中提取值得跨会话记住的用户健康档案事实。
分类固定为：allergy(过敏史) / medication(在用药品) / chronic(慢病) / history(既往史、手术史) /
preference(沟通与就诊偏好) / basic(基本情况) / other。
只提取明确出现的信息；与已有档案重复的不要提取。同时用一句话概括本次会话主题。
已有档案：{existing_ltm}

会话摘要：{summary}

会话记录：
{transcript}"""
