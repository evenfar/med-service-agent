"""医疗模拟数据（第3期）：全部为教学虚构，非真实药品/医疗建议。

设计原则（沿用 ecom 的思路）：数据状态要"成对覆盖"——
预约覆盖 各状态、药品覆盖 OTC/处方、相互作用覆盖 无/中/高 风险、
报告覆盖 正常/偏高/危急，让每条工具分支都有可触发的用例。
"""

from __future__ import annotations

import copy

# ---------------- 挂号预约记录 ----------------

_APPOINTMENTS_SEED = {
    "GH-2026-001": {
        "appointment_id": "GH-2026-001", "department": "呼吸内科",
        "doctor": "陈志远", "date": "2026-09-25", "time": "09:30",
        "status": "booked", "patient": "张先生", "fee": 50.0,
    },
    "GH-2026-002": {
        "appointment_id": "GH-2026-002", "department": "心血管内科",
        "doctor": "李文娟", "date": "2026-09-24", "time": "14:00",
        "status": "pending_payment", "patient": "张先生", "fee": 80.0,
    },
    "GH-2026-003": {
        "appointment_id": "GH-2026-003", "department": "皮肤科",
        "doctor": "王雅静", "date": "2026-09-18", "time": "10:00",
        "status": "completed", "patient": "张先生", "fee": 30.0,
    },
    "GH-2026-004": {
        "appointment_id": "GH-2026-004", "department": "儿科",
        "doctor": "刘敏", "date": "2026-09-20", "time": "15:30",
        "status": "cancelled", "patient": "张先生", "fee": 60.0,
    },
    "GH-2026-005": {
        "appointment_id": "GH-2026-005", "department": "消化内科",
        "doctor": "赵国梁", "date": "2026-09-26", "time": "11:00",
        "status": "booked", "patient": "张先生", "fee": 50.0,
    },
}

# ---------------- 药品目录 ----------------

MEDICINES = {
    "布洛芬": {
        "name": "布洛芬", "type": "OTC", "category": "解热镇痛",
        "indication": "用于缓解轻至中度疼痛及发热",
        "dosage": "成人一次0.3g，一日2次，随餐服用（详见说明书）",
        "precautions": "消化道溃疡者慎用；不宜与其他解热镇痛药同服；孕妇遵医嘱",
    },
    "对乙酰氨基酚": {
        "name": "对乙酰氨基酚", "type": "OTC", "category": "解热镇痛",
        "indication": "用于退热及缓解头痛、牙痛等轻中度疼痛",
        "dosage": "成人一次0.5g，一日不超过4次（详见说明书）",
        "precautions": "每日总量不超过2g；服药期间避免饮酒；肝功能不全者慎用",
    },
    "阿莫西林": {
        "name": "阿莫西林", "type": "处方药", "category": "抗生素",
        "indication": "用于敏感菌引起的感染（须医生处方）",
        "dosage": "遵医嘱，常见疗程7~10天，不可自行停药",
        "precautions": "青霉素过敏者禁用；不可自行购买使用",
    },
    "氯雷他定": {
        "name": "氯雷他定", "type": "OTC", "category": "抗过敏",
        "indication": "缓解过敏性鼻炎、慢性荨麻疹等症状",
        "dosage": "成人一日1次，一次10mg",
        "precautions": "驾驶前注意个体嗜睡反应；肝功能不全者减量",
    },
    "二甲双胍": {
        "name": "二甲双胍", "type": "处方药", "category": "降糖药",
        "indication": "2型糖尿病血糖控制（须医生处方）",
        "dosage": "遵医嘱，一般随餐服用",
        "precautions": "肾功能不全者慎用；造影检查前后需暂停；不可自行调整剂量",
    },
    "华法林": {
        "name": "华法林", "type": "处方药", "category": "抗凝药",
        "indication": "预防及治疗血栓（须医生处方，定期监测INR）",
        "dosage": "严格遵医嘱，按INR监测结果调整",
        "precautions": "与多种药物/食物相互作用；避免大量绿叶菜波动摄入；出血倾向立即就医",
    },
    "阿司匹林": {
        "name": "阿司匹林", "type": "处方药", "category": "抗血小板",
        "indication": "心脑血管疾病二级预防（须医生评估）",
        "dosage": "遵医嘱，常见小剂量长期服用",
        "precautions": "出血风险者慎用；胃部不适及时就医；手术前须告知医生",
    },
    "硝酸甘油": {
        "name": "硝酸甘油", "type": "处方药", "category": "抗心绞痛",
        "indication": "冠心病心绞痛急性发作时舌下含服（须医生处方）",
        "dosage": "发作时舌下含服1片，5分钟不减缓解可重复，最多3次",
        "precautions": "含服3次无效立即拨打120；禁与西地那非类同用；避光保存",
    },
}

# ---------------- 药物相互作用表（按字母序对存储） ----------------

DRUG_INTERACTIONS = {
    ("华法林", "阿司匹林"): {
        "severity": "高风险", "effect": "抗凝+抗血小板叠加，出血风险显著升高",
        "advice": "两药联用须医生严格评估并监测凝血指标，出现瘀斑、黑便、牙龈出血立即就医",
    },
    ("布洛芬", "阿司匹林"): {
        "severity": "中风险", "effect": "胃肠道刺激叠加，消化道出血风险升高",
        "advice": "避免同服；确需镇痛建议咨询医生换用对乙酰氨基酚",
    },
    ("布洛芬", "华法林"): {
        "severity": "中风险", "effect": "非甾体抗炎药可增强抗凝作用",
        "advice": "服用华法林期间避免自行使用布洛芬，镇痛请先咨询医生",
    },
    ("二甲双胍", "华法林"): {
        "severity": "低风险", "effect": "个别报道影响抗凝效果",
        "advice": "同用期间遵医嘱加强INR监测即可",
    },
}

# ---------------- 检验报告 ----------------

LAB_REPORTS = {
    "LAB-2026-001": {
        "report_id": "LAB-2026-001", "name": "血常规", "date": "2026-09-15",
        "conclusion": "轻度贫血倾向，建议复查并就诊血液内科或营养科",
        "items": [
            {"name": "白细胞计数", "value": 6.2, "unit": "×10⁹/L", "ref": "3.5-9.5", "flag": "正常"},
            {"name": "血红蛋白", "value": 112, "unit": "g/L", "ref": "130-175", "flag": "偏低"},
            {"name": "血小板计数", "value": 245, "unit": "×10⁹/L", "ref": "125-350", "flag": "正常"},
        ],
    },
    "LAB-2026-002": {
        "report_id": "LAB-2026-002", "name": "肝功能", "date": "2026-09-16",
        "conclusion": "转氨酶轻度升高，建议清淡饮食一周后复查，持续升高就诊消化内科",
        "items": [
            {"name": "谷丙转氨酶ALT", "value": 68, "unit": "U/L", "ref": "9-50", "flag": "偏高"},
            {"name": "谷草转氨酶AST", "value": 45, "unit": "U/L", "ref": "15-40", "flag": "偏高"},
            {"name": "总胆红素", "value": 15.1, "unit": "μmol/L", "ref": "3.4-17.1", "flag": "正常"},
        ],
    },
    "LAB-2026-003": {
        "report_id": "LAB-2026-003", "name": "空腹血糖+血脂", "date": "2026-09-17",
        "conclusion": "空腹血糖受损伴血脂偏高，建议内分泌科就诊并调整饮食运动",
        "items": [
            {"name": "空腹血糖", "value": 6.8, "unit": "mmol/L", "ref": "3.9-6.1", "flag": "偏高"},
            {"name": "总胆固醇", "value": 5.9, "unit": "mmol/L", "ref": "<5.2", "flag": "偏高"},
            {"name": "低密度脂蛋白", "value": 3.8, "unit": "mmol/L", "ref": "<3.4", "flag": "偏高"},
        ],
    },
    "LAB-2026-004": {
        "report_id": "LAB-2026-004", "name": "电解质", "date": "2026-09-21",
        "conclusion": "血钾危急值！已触发危急值报告流程，请立即联系开单医生或前往急诊",
        "items": [
            {"name": "血钾", "value": 2.6, "unit": "mmol/L", "ref": "3.5-5.3", "flag": "危急"},
            {"name": "血钠", "value": 138, "unit": "mmol/L", "ref": "137-147", "flag": "正常"},
        ],
    },
}

# ---------------- 科室表（导诊用） ----------------

DEPARTMENTS = {
    "呼吸内科": {"name": "呼吸内科", "description": "咳嗽、咳痰、哮喘、肺部结节随访",
                 "keywords": ["咳嗽", "咳痰", "气短", "哮喘", "感冒", "肺部"]},
    "心血管内科": {"name": "心血管内科", "description": "胸闷、心悸、高血压、冠心病随访",
                   "keywords": ["心悸", "胸闷", "高血压", "冠心", "心跳"]},
    "消化内科": {"name": "消化内科", "description": "腹痛、腹泻、反酸、肝功能异常随访",
                 "keywords": ["腹痛", "腹泻", "反酸", "胃", "肝", "便秘"]},
    "内分泌科": {"name": "内分泌科", "description": "糖尿病、甲状腺疾病、血糖血脂异常",
                 "keywords": ["血糖", "糖尿", "甲状腺", "血脂", "多饮", "多尿"]},
    "儿科": {"name": "儿科", "description": "14岁以下儿童常见病诊疗",
             "keywords": ["儿童", "小孩", "宝宝", "婴儿"]},
    "皮肤科": {"name": "皮肤科", "description": "皮疹、湿疹、荨麻疹、脱发",
               "keywords": ["皮疹", "湿疹", "荨麻疹", "皮肤", "脱发", "瘙痒"]},
    "神经内科": {"name": "神经内科", "description": "头痛、头晕、失眠、肢体麻木",
                 "keywords": ["头痛", "头晕", "失眠", "麻木", "手抖"]},
    "急诊科": {"name": "急诊科", "description": "急危重症24小时接诊",
               "keywords": ["急诊", "急救"]},
}

# ---------------- 可变状态管理 ----------------
# 取消/改期会修改预约状态（mock 的"写操作"），为避免测试间串扰提供深拷贝重置。

APPOINTMENTS: dict[str, dict] = {}


def reset_mock_data() -> None:
    """恢复全部模拟数据到初始状态（测试隔离用）。"""
    APPOINTMENTS.clear()
    for k, v in _APPOINTMENTS_SEED.items():
        APPOINTMENTS[k] = copy.deepcopy(v)


reset_mock_data()
