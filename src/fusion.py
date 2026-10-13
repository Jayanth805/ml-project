"""
Multimodal fusion module for combining text, image, and consistency features.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for the fusion module."""
    text_embedding_dim: int = 768
    image_embedding_dim: int = 768
    projection_dim: int = 256
    fusion_hidden_dim: int = 256
    dropout: float = 0.3
    output_dim: int = 1
    include_consistency: bool = True


class FeatureProjection(nn.Module):
    """
    Trainable projection layers for text and image embeddings.
    Projects high-dimensional embeddings to a lower-dimensional space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize projection layer.

        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply projection."""
        return self.projection(x)


class ConsistencyAwareFusion(nn.Module):
    """
    Consistency-aware multimodal fusion module.

    Combines:
    1. Projected text features (T_proj)
    2. Projected image features (I_proj)
    3. Absolute difference |T_proj - I_proj|
    4. Element-wise product T_proj * I_proj
    5. CLIP semantic similarity score

    The fusion vector Z has dimension: 4 * projection_dim + 1
    """

    def __init__(self, config: FusionConfig):
        """
        Initialize fusion module.

        Args:
            config: Fusion configuration
        """
        super().__init__()

        self.config = config

        # Projection layers
        self.text_projection = FeatureProjection(
            input_dim=config.text_embedding_dim,
            output_dim=config.projection_dim,
            dropout=config.dropout
        )

        self.image_projection = FeatureProjection(
            input_dim=config.image_embedding_dim,
            output_dim=config.projection_dim,
            dropout=config.dropout
        )

        # Fusion input dimension
        # T_proj (256) + I_proj (256) + |T_proj - I_proj| (256) + T_proj * I_proj (256) + CLIP_sim (1)
        fusion_input_dim = 4 * config.projection_dim + 1

        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim // 2, config.output_dim)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        clip_similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through fusion module.

        Args:
            text_features: Text embeddings [batch_size, text_embedding_dim]
            image_features: Image embeddings [batch_size, image_embedding_dim]
            clip_similarity: CLIP cosine similarities [batch_size]

        Returns:
            Tuple of (logits, intermediate_outputs)
        """
        # Project to common space
        text_proj = self.text_projection(text_features)  # [batch_size, projection_dim]
        image_proj = self.image_projection(image_features)  # [batch_size, projection_dim]

        # Compute interaction features
        abs_diff = torch.abs(text_proj - image_proj)  # [batch_size, projection_dim]
        hadamard_product = text_proj * image_proj  # [batch_size, projection_dim]

        # Prepare CLIP similarity [batch_size, 1]
        clip_sim = clip_similarity.unsqueeze(-1)

        # Concatenate all features
        fusion_vector = torch.cat([
            text_proj,
            image_proj,
            abs_diff,
            hadamard_product,
            clip_sim
        ], dim=-1)  # [batch_size, 4*projection_dim + 1]

        # Classification
        logits = self.fusion_classifier(fusion_vector)  # [batch_size, 1]

        # Store intermediate outputs for analysis
        intermediate = {
            'text_proj': text_proj,
            'image_proj': image_proj,
            'abs_diff': abs_diff,
            'hadamard_product': hadamard_product,
            'fusion_vector': fusion_vector,
            'clip_similarity': clip_similarity
        }

        return logits, intermediate

    def get_fusion_dim(self) -> int:
        """Get the dimension of the fusion vector."""
        return 4 * self.config.projection_dim + 1


class StandardFusion(nn.Module):
    """
    Standard multimodal fusion without consistency features.
    Used as baseline for comparison.

    Combines:
    1. Projected text features
    2. Projected image features
    3. Concatenation only (no difference/product/similarity)
    """

    def __init__(
        self,
        text_embedding_dim: int = 768,
        image_embedding_dim: int = 768,
        projection_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        """
        Initialize standard fusion module.

        Args:
            text_embedding_dim: Text embedding dimension
            image_embedding_dim: Image embedding dimension
            projection_dim: Projection dimension
            fusion_hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            output_dim: Output dimension (1 for binary)
        """
        super().__init__()

        # Projection layers
        self.text_projection = FeatureProjection(
            input_dim=text_embedding_dim,
            output_dim=projection_dim,
            dropout=dropout
        )

        self.image_projection = FeatureProjection(
            input_dim=image_embedding_dim,
            output_dim=projection_dim,
            dropout=dropout
        )

        # Standard fusion: concatenation of projections
        fusion_input_dim = 2 * projection_dim

        # Fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through standard fusion.

        Args:
            text_features: Text embeddings [batch_size, text_embedding_dim]
            image_features: Image embeddings [batch_size, image_embedding_dim]

        Returns:
            Tuple of (logits, intermediate_outputs)
        """
        # Project to common space
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)

        # Simple concatenation
        fusion_vector = torch.cat([text_proj, image_proj], dim=-1)

        # Classification
        logits = self.fusion_classifier(fusion_vector)

        intermediate = {
            'text_proj': text_proj,
            'image_proj': image_proj,
            'fusion_vector': fusion_vector
        }

        return logits, intermediate


class TextOnlyClassifier(nn.Module):
    """Text-only baseline classifier."""

    def __init__(
        self,
        text_embedding_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, text_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logits = self.classifier(text_features)
        return logits, {'text_features': text_features}


class ImageOnlyClassifier(nn.Module):
    """Image-only baseline classifier."""

    def __init__(
        self,
        image_embedding_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logits = self.classifier(image_features)
        return logits, {'image_features': image_features}


class AblationFusion(nn.Module):
    """
    Ablation model: Consistency-aware fusion WITHOUT CLIP similarity.
    Tests the contribution of explicit consistency features.
    """

    def __init__(
        self,
        text_embedding_dim: int = 768,
        image_embedding_dim: int = 768,
        projection_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        output_dim: int = 1
    ):
        """
        Initialize ablation fusion module.

        Same as consistency-aware but without CLIP similarity.
        """
        super().__init__()

        # Projection layers
        self.text_projection = FeatureProjection(
            input_dim=text_embedding_dim,
            output_dim=projection_dim,
            dropout=dropout
        )

        self.image_projection = FeatureProjection(
            input_dim=image_embedding_dim,
            output_dim=projection_dim,
            dropout=dropout
        )

        # Fusion WITHOUT CLIP similarity
        fusion_input_dim = 3 * projection_dim  # T_proj + I_proj + |T - I|

        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, output_dim)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        clip_similarity: Optional[torch.Tensor] = None  # Ignored in ablation
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass (CLIP similarity is computed but not used).
        """
        # Project to common space
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)

        # Compute difference only (no similarity, no product)
        abs_diff = torch.abs(text_proj - image_proj)

        # Concatenation WITHOUT similarity
        fusion_vector = torch.cat([text_proj, image_proj, abs_diff], dim=-1)

        # Classification
        logits = self.fusion_classifier(fusion_vector)

        intermediate = {
            'text_proj': text_proj,
            'image_proj': image_proj,
            'abs_diff': abs_diff,
            'fusion_vector': fusion_vector,
            'clip_similarity': clip_similarity
        }

        return logits, intermediate


def create_fusion_module(
    model_type: str,
    config: Optional[FusionConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion modules.

    Args:
        model_type: One of 'consistency_aware', 'standard', 'text_only',
                   'image_only', 'ablation'
        config: Optional fusion configuration
        **kwargs: Additional arguments for specific module types

    Returns:
        Initialized fusion module
    """
    if config is None:
        config = FusionConfig(**kwargs)

    if model_type == 'consistency_aware':
        return ConsistencyAwareFusion(config)
    elif model_type == 'standard':
        return StandardFusion(
            text_embedding_dim=config.text_embedding_dim,
            image_embedding_dim=config.image_embedding_dim,
            projection_dim=config.projection_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
            dropout=config.dropout,
            output_dim=config.output_dim
        )
    elif model_type == 'text_only':
        return TextOnlyClassifier(
            text_embedding_dim=config.text_embedding_dim,
            hidden_dim=config.fusion_hidden_dim,
            dropout=config.dropout,
            output_dim=config.output_dim
        )
    elif model_type == 'image_only':
        return ImageOnlyClassifier(
            image_embedding_dim=config.image_embedding_dim,
            hidden_dim=config.fusion_hidden_dim,
            dropout=config.dropout,
            output_dim=config.output_dim
        )
    elif model_type == 'ablation':
        return AblationFusion(
            text_embedding_dim=config.text_embedding_dim,
            image_embedding_dim=config.image_embedding_dim,
            projection_dim=config.projection_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
            dropout=config.dropout,
            output_dim=config.output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
