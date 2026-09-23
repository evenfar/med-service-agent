"""Schema 与配置测试（第0/1期）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.settings import Settings, is_offline
from app.schemas.response import INTENT_LABELS, IntentType, MedicalResponse, UrgencyLevel


class TestSettings:
    def test_defaults_offline_when_no_key(self):
        s = Settings(mock_mode=False, openai_api_key="")
        assert is_offline(s) is True

    def test_explicit_mock(self):
        assert is_offline(Settings(mock_mode=True)) is True

    def test_kwargs_override_env(self, tmp_path):
        s = Settings(mock_mode=True, session_path=str(tmp_path / "s.json"))
        assert s.session_path.endswith("s.json")
        assert s.max_react_steps == 6  # 编排护栏默认存在

    def test_multi_instance_isolation(self):
        """两个 Settings 实例互不影响（评估沙箱依赖这一点）。"""
        a = Settings(mock_mode=False, memory_enabled=True)
        b = Settings(mock_mode=True, memory_enabled=False)
        assert a.memory_enabled and not b.memory_enabled


class TestMedicalResponse:
    def test_minimal_valid(self):
        r = MedicalResponse(reply="你好")
        assert r.intent == IntentType.other
        assert r.requires_human is False
        assert r.urgency == UrgencyLevel.routine

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            MedicalResponse(reply="x", confidence=1.5)
        with pytest.raises(ValidationError):
            MedicalResponse(reply="x", confidence=-0.1)

    def test_reply_required(self):
        with pytest.raises(ValidationError):
            MedicalResponse(reply="")

    def test_intent_labels_cover_all(self):
        assert set(INTENT_LABELS) == set(IntentType)

    def test_serialization_roundtrip(self):
        r = MedicalResponse(reply="测试", intent=IntentType.emergency,
                            confidence=0.9, requires_human=True,
                            urgency=UrgencyLevel.emergency)
        r2 = MedicalResponse.model_validate_json(r.model_dump_json())
        assert r2 == r
