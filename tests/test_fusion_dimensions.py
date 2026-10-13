"""
Tests for fusion module dimensions.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import (
    ConsistencyAwareFusion,
    StandardFusion,
    TextOnlyClassifier,
    ImageOnlyClassifier,
    AblationFusion,
    FusionConfig,
    create_fusion_module
)


class TestFusionDimensions:
    """Test fusion module dimensions."""

    def test_consistency_aware_fusion_dimensions(self):
        """Test consistency-aware fusion produces correct output dimensions."""
        config = FusionConfig(
            text_embedding_dim=768,
            image_embedding_dim=768,
            projection_dim=256,
            fusion_hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )

        model = ConsistencyAwareFusion(config)
        model.eval()

        # Create dummy inputs
        batch_size = 4
        text_features = torch.randn(batch_size, 768)
        image_features = torch.randn(batch_size, 768)
        clip_similarity = torch.randn(batch_size)

        # Forward pass
        logits, intermediate = model(text_features, image_features, clip_similarity)

        # Check output shape
        assert logits.shape == (batch_size, 1)

        # Check intermediate shapes
        assert intermediate['text_proj'].shape == (batch_size, 256)
        assert intermediate['image_proj'].shape == (batch_size, 256)
        assert intermediate['abs_diff'].shape == (batch_size, 256)
        assert intermediate['hadamard_product'].shape == (batch_size, 256)
        assert intermediate['clip_similarity'].shape == (batch_size,)

    def test_standard_fusion_dimensions(self):
        """Test standard fusion produces correct output dimensions."""
        model = StandardFusion(
            text_embedding_dim=768,
            image_embedding_dim=768,
            projection_dim=256,
            fusion_hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )
        model.eval()

        batch_size = 4
        text_features = torch.randn(batch_size, 768)
        image_features = torch.randn(batch_size, 768)

        logits, intermediate = model(text_features, image_features)

        assert logits.shape == (batch_size, 1)
        assert intermediate['text_proj'].shape == (batch_size, 256)
        assert intermediate['image_proj'].shape == (batch_size, 256)

    def test_text_only_classifier_dimensions(self):
        """Test text-only classifier produces correct dimensions."""
        model = TextOnlyClassifier(
            text_embedding_dim=768,
            hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )
        model.eval()

        batch_size = 4
        text_features = torch.randn(batch_size, 768)

        logits, intermediate = model(text_features)

        assert logits.shape == (batch_size, 1)
        assert 'text_features' in intermediate

    def test_image_only_classifier_dimensions(self):
        """Test image-only classifier produces correct dimensions."""
        model = ImageOnlyClassifier(
            image_embedding_dim=768,
            hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )
        model.eval()

        batch_size = 4
        image_features = torch.randn(batch_size, 768)

        logits, intermediate = model(image_features)

        assert logits.shape == (batch_size, 1)
        assert 'image_features' in intermediate

    def test_ablation_fusion_dimensions(self):
        """Test ablation fusion produces correct dimensions (no CLIP similarity)."""
        model = AblationFusion(
            text_embedding_dim=768,
            image_embedding_dim=768,
            projection_dim=256,
            fusion_hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )
        model.eval()

        batch_size = 4
        text_features = torch.randn(batch_size, 768)
        image_features = torch.randn(batch_size, 768)
        clip_similarity = torch.randn(batch_size)  # Should be ignored

        logits, intermediate = model(text_features, image_features, clip_similarity)

        assert logits.shape == (batch_size, 1)
        # Ablation uses 3 * projection_dim = 768 input to classifier
        assert intermediate['fusion_vector'].shape == (batch_size, 768)  # 256*3

    def test_create_fusion_module_factory(self):
        """Test the factory function creates correct modules."""
        config = FusionConfig(
            text_embedding_dim=768,
            image_embedding_dim=768,
            projection_dim=256,
            fusion_hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )

        # Test each model type
        for model_type in ['consistency_aware', 'standard', 'text_only', 'image_only', 'ablation']:
            model = create_fusion_module(model_type, config)
            assert model is not None
            assert hasattr(model, 'forward')

    def test_fusion_vector_dimension(self):
        """Test that fusion vector has expected dimension."""
        config = FusionConfig(
            text_embedding_dim=768,
            image_embedding_dim=768,
            projection_dim=256,
            fusion_hidden_dim=256,
            dropout=0.3,
            output_dim=1
        )

        model = ConsistencyAwareFusion(config)

        # Check dimension calculation
        expected_dim = 4 * 256 + 1  # T, I, diff, product, similarity
        assert model.get_fusion_dim() == expected_dim

        # Verify through forward pass
        batch_size = 2
        text_features = torch.randn(batch_size, 768)
        image_features = torch.randn(batch_size, 768)
        clip_similarity = torch.randn(batch_size)

        _, intermediate = model(text_features, image_features, clip_similarity)
        assert intermediate['fusion_vector'].shape == (batch_size, expected_dim)


class TestFusionGradients:
    """Test that fusion modules have trainable parameters."""

    def test_projection_layers_trainable(self):
        """Test that projection layers are trainable."""
        model = ConsistencyAwareFusion(FusionConfig())
        model.train()

        # Check that projection parameters require gradients
        for name, param in model.named_parameters():
            if 'projection' in name:
                assert param.requires_grad, f"Parameter {name} should be trainable"

    def test_classifier_trainable(self):
        """Test that classifier layers are trainable."""
        model = ConsistencyAwareFusion(FusionConfig())
        model.train()

        # Check classifier parameters
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
        assert len(classifier_params) > 0
        for param in classifier_params:
            assert param.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
