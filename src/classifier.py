"""
Classifier module that wraps the complete model architecture.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

from .encoders import EncoderEnsemble, EncoderConfig
from .fusion import FusionConfig, ConsistencyAwareFusion, StandardFusion, create_fusion_module


@dataclass
class ClassifierConfig:
    """Configuration for the complete classifier."""
    # Encoder settings
    text_encoder_name: str = "google-bert/bert-base-uncased"
    image_encoder_name: str = "google/vit-base-patch16-224"
    clip_name: str = "openai/clip-vit-base-patch32"
    text_embedding_dim: int = 768
    image_embedding_dim: int = 768
    clip_embedding_dim: int = 512
    freeze_encoders: bool = True

    # Fusion settings
    projection_dim: int = 256
    fusion_hidden_dim: int = 256
    dropout: float = 0.3
    output_dim: int = 1

    # Model type
    model_type: str = "consistency_aware"  # consistency_aware, standard, text_only, image_only, ablation


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal fake news classifier.

    Combines:
    - Encoder ensemble (BERT, ViT, CLIP)
    - Fusion module
    - Classification head
    """

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the classifier.

        Args:
            config: Classifier configuration
        """
        super().__init__()

        self.config = config
        self.model_type = config.model_type

        # Create encoder ensemble
        encoder_config = EncoderConfig(
            text_encoder_name=config.text_encoder_name,
            image_encoder_name=config.image_encoder_name,
            clip_name=config.clip_name,
            text_embedding_dim=config.text_embedding_dim,
            image_embedding_dim=config.image_embedding_dim,
            clip_embedding_dim=config.clip_embedding_dim,
            freeze=config.freeze_encoders
        )

        self.encoders = EncoderEnsemble(config=encoder_config)

        # Create fusion module based on model type
        if config.model_type == 'consistency_aware':
            fusion_config = FusionConfig(
                text_embedding_dim=config.text_embedding_dim,
                image_embedding_dim=config.image_embedding_dim,
                projection_dim=config.projection_dim,
                fusion_hidden_dim=config.fusion_hidden_dim,
                dropout=config.dropout,
                output_dim=config.output_dim,
                include_consistency=True
            )
            self.fusion = ConsistencyAwareFusion(fusion_config)
        elif config.model_type == 'standard':
            self.fusion = StandardFusion(
                text_embedding_dim=config.text_embedding_dim,
                image_embedding_dim=config.image_embedding_dim,
                projection_dim=config.projection_dim,
                fusion_hidden_dim=config.fusion_hidden_dim,
                dropout=config.dropout,
                output_dim=config.output_dim
            )
        elif config.model_type == 'text_only':
            self.fusion = None
            self.text_classifier = nn.Sequential(
                nn.Linear(config.text_embedding_dim, config.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim // 2, config.output_dim)
            )
        elif config.model_type == 'image_only':
            self.fusion = None
            self.image_classifier = nn.Sequential(
                nn.Linear(config.image_embedding_dim, config.fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim // 2, config.output_dim)
            )
        elif config.model_type == 'ablation':
            self.fusion = create_fusion_module('ablation', None,
                text_embedding_dim=config.text_embedding_dim,
                image_embedding_dim=config.image_embedding_dim,
                projection_dim=config.projection_dim,
                fusion_hidden_dim=config.fusion_hidden_dim,
                dropout=config.dropout,
                output_dim=config.output_dim
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor = None,
        clip_input_ids: torch.Tensor = None,
        clip_attention_mask: torch.Tensor = None,
        clip_pixel_values: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the complete model.

        Args:
            text_input_ids: BERT text tokens
            text_attention_mask: BERT attention mask
            pixel_values: ViT image pixels
            clip_input_ids: CLIP text tokens
            clip_attention_mask: CLIP text attention mask
            clip_pixel_values: CLIP image pixels

        Returns:
            Tuple of (logits, intermediate_outputs)
        """
        if self.model_type == 'text_only':
            if text_input_ids is None or text_attention_mask is None:
                raise ValueError("The text-only model requires tokenized text")
            # Extract text features only
            text_features = self.encoders.extract_text_features(
                text_input_ids, text_attention_mask
            )
            logits = self.text_classifier(text_features)
            intermediate = {'text_features': text_features}
            return logits, intermediate

        elif self.model_type == 'image_only':
            if pixel_values is None:
                raise ValueError("The image-only model requires image pixels")
            # Extract image features only
            image_features = self.encoders.extract_image_features(pixel_values)
            logits = self.image_classifier(image_features)
            intermediate = {'image_features': image_features}
            return logits, intermediate

        else:
            # Multimodal models
            if text_input_ids is None or text_attention_mask is None:
                raise ValueError(f"The {self.model_type} model requires tokenized text")
            if pixel_values is None:
                raise ValueError(f"The {self.model_type} model requires image pixels")
            text_features = self.encoders.extract_text_features(
                text_input_ids, text_attention_mask
            )
            image_features = self.encoders.extract_image_features(pixel_values)

            if self.model_type == 'consistency_aware':
                if clip_input_ids is None or clip_attention_mask is None or clip_pixel_values is None:
                    raise ValueError("The consistency-aware model requires CLIP text and image inputs")
                clip_similarity = self.encoders.compute_clip_similarity(
                    clip_input_ids, clip_attention_mask, clip_pixel_values
                )
                logits, intermediate = self.fusion(
                    text_features, image_features, clip_similarity
                )
            elif self.model_type == 'standard':
                logits, intermediate = self.fusion(
                    text_features, image_features
                )
            elif self.model_type == 'ablation':
                logits, intermediate = self.fusion(
                    text_features, image_features, None
                )

            return logits, intermediate

    def predict_proba(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor = None,
        clip_input_ids: torch.Tensor = None,
        clip_attention_mask: torch.Tensor = None,
        clip_pixel_values: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction probabilities.

        Args:
            Same as forward()

        Returns:
            Tuple of (prob_fake, prob_real) - both [batch_size]
        """
        logits, _ = self.forward(
            text_input_ids, text_attention_mask, pixel_values,
            clip_input_ids, clip_attention_mask, clip_pixel_values
        )

        probs = torch.sigmoid(logits.squeeze(-1))
        return probs, 1 - probs

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': self.model_type,
            'text_encoder': self.config.text_encoder_name,
            'image_encoder': self.config.image_encoder_name,
            'clip_encoder': self.config.clip_name,
            'projection_dim': self.config.projection_dim,
            'fusion_hidden_dim': self.config.fusion_hidden_dim,
            'dropout': self.config.dropout,
            'freeze_encoders': self.config.freeze_encoders
        }


def create_classifier(
    model_type: str,
    config: Optional[ClassifierConfig] = None,
    **kwargs
) -> MultimodalClassifier:
    """
    Factory function to create classifiers.

    Args:
        model_type: Model type (consistency_aware, standard, text_only, image_only, ablation)
        config: Optional classifier configuration
        **kwargs: Additional configuration parameters

    Returns:
        Initialized classifier
    """
    if config is None:
        if kwargs:
            config = ClassifierConfig(model_type=model_type, **kwargs)
        else:
            config = ClassifierConfig(model_type=model_type)

    return MultimodalClassifier(config)
