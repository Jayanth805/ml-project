"""Checks for the multimodal low-alignment decision rule."""

import pytest

from src.utils import apply_mismatch_policy


@pytest.mark.parametrize("model_type", ["standard", "consistency_aware", "ablation"])
def test_low_clip_similarity_overrides_real_for_multimodal_models(model_type):
    result = apply_mismatch_policy("REAL", model_type, 0.15)

    assert result["prediction"] == "FAKE"
    assert result["classifier_prediction"] == "REAL"
    assert result["prediction_reason"] == "low_clip_alignment_override"
    assert result["possible_mismatch"] is True


@pytest.mark.parametrize("model_type", ["standard", "consistency_aware", "ablation"])
def test_high_clip_similarity_preserves_classifier_result(model_type):
    result = apply_mismatch_policy("REAL", model_type, 0.30)

    assert result["prediction"] == "REAL"
    assert result["prediction_reason"] == "classifier"
    assert result["possible_mismatch"] is False


def test_rule_does_not_change_single_modality_models():
    result = apply_mismatch_policy("REAL", "text_only", 0.10)

    assert result["prediction"] == "REAL"
    assert result["possible_mismatch"] is False


def test_missing_clip_similarity_does_not_override():
    result = apply_mismatch_policy("REAL", "standard", None)

    assert result["prediction"] == "REAL"
    assert result["possible_mismatch"] is False
