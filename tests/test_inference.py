"""
Tests for inference module.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import NewsAnalyzer, PredictionResult
from src.classifier import MultimodalClassifier, ClassifierConfig


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test creating a prediction result."""
        result = PredictionResult(
            prediction="FAKE",
            probability_fake=0.75,
            probability_real=0.25,
            confidence=0.75,
            clip_similarity=0.85,
            consistency_interpretation={
                "raw_similarity": 0.85,
                "interpretation": "Strong semantic match",
                "level": "high",
                "warning": "Semantic similarity does not indicate factual accuracy"
            },
            model_type="consistency_aware",
            model_version="1.0.0",
            timestamp="2024-01-01T00:00:00",
            text_preview="Test headline...",
            image_filename="test.jpg"
        )

        assert result.prediction == "FAKE"
        assert result.probability_fake == 0.75
        assert result.probability_real == 0.25
        assert result.confidence == 0.75
        assert result.clip_similarity == 0.85
        assert result.consistency_interpretation["level"] == "high"


class TestNewsAnalyzer:
    """Tests for NewsAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test initializing the analyzer without checkpoint."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        assert analyzer is not None
        assert analyzer.model is not None
        assert analyzer.preprocessor is not None

    def test_analyzer_empty_text_raises_error(self):
        """Test that empty text raises an error."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        with pytest.raises(ValueError):
            analyzer.analyze("")

    def test_analyzer_whitespace_text_raises_error(self):
        """Test that whitespace-only text raises an error."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        with pytest.raises(ValueError):
            analyzer.analyze("   ")

    def test_analyzer_text_only_analysis(self):
        """Test analyzing text without image."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        result = analyzer.analyze("This is a test news headline about something important.")

        assert result is not None
        assert result.prediction in ["FAKE", "REAL"]
        assert 0 <= result.probability_fake <= 1
        assert 0 <= result.probability_real <= 1
        assert result.text_preview is not None

    def test_checkpoint_loaded_property(self):
        """Test checkpoint_loaded property."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        # Should be False without checkpoint
        assert analyzer.checkpoint_loaded == False

    def test_is_ready_property(self):
        """Test is_ready property."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        # Should be False without trained checkpoint
        assert analyzer.is_ready() == False


class TestModelInference:
    """Tests for model inference behavior."""

    def test_model_output_shape(self):
        """Test that model produces correct output shape."""
        config = ClassifierConfig(model_type="consistency_aware")
        model = MultimodalClassifier(config)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        clip_input_ids = torch.randint(0, 1000, (batch_size, 77))
        clip_attention_mask = torch.ones(batch_size, 77)
        clip_pixel_values = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits, intermediate = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                pixel_values=pixel_values,
                clip_input_ids=clip_input_ids,
                clip_attention_mask=clip_attention_mask,
                clip_pixel_values=clip_pixel_values
            )

        assert logits.shape == (batch_size, 1)

    def test_text_only_model(self):
        """Test text-only model."""
        config = ClassifierConfig(model_type="text_only")
        model = MultimodalClassifier(config)
        model.eval()

        batch_size = 2
        text_input_ids = torch.randint(0, 1000, (batch_size, 128))
        text_attention_mask = torch.ones(batch_size, 128)
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        clip_input_ids = torch.randint(0, 1000, (batch_size, 77))
        clip_attention_mask = torch.ones(batch_size, 77)
        clip_pixel_values = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits, intermediate = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                pixel_values=pixel_values,
                clip_input_ids=clip_input_ids,
                clip_attention_mask=clip_attention_mask,
                clip_pixel_values=clip_pixel_values
            )

        assert logits.shape == (batch_size, 1)
        assert 'text_features' in intermediate


class TestInputValidation:
    """Tests for input validation."""

    def test_long_text_truncation(self):
        """Test that long text is handled."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        # Create very long text
        long_text = "Test " * 1000

        result = analyzer.analyze(long_text)

        assert result is not None
        assert result.text_preview is not None

    def test_special_characters_in_text(self):
        """Test that special characters are handled."""
        analyzer = NewsAnalyzer(model_type="consistency_aware")

        text = "Test with special chars: @#$%^&*()_+-={}[]|\\:\";'<>?,./~`"

        result = analyzer.analyze(text)

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
