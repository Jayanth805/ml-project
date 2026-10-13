"""
Encoder modules for text and image processing.
Contains BERT, ViT, and CLIP encoders.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
    BertModel,
    ViTModel
)
from PIL import Image


@dataclass
class EncoderConfig:
    """Configuration for encoder models."""
    text_encoder_name: str = "google-bert/bert-base-uncased"
    image_encoder_name: str = "google/vit-base-patch16-224"
    clip_name: str = "openai/clip-vit-base-patch32"
    text_embedding_dim: int = 768
    image_embedding_dim: int = 768
    clip_embedding_dim: int = 512
    projection_dim: int = 256
    freeze: bool = True
    device: str = "cuda"


class BERTEncoder(nn.Module):
    """
    BERT-based text encoder.
    Extracts CLS token representation from BERT.
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        embedding_dim: int = 768,
        freeze: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize BERT encoder.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Output embedding dimension
            freeze: Whether to freeze pretrained weights
            device: Device to use
        """
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim

        print(f"Loading BERT model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze:
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False

        # Verify embedding dimension matches
        actual_dim = self.bert.config.hidden_size
        if actual_dim != embedding_dim:
            print(f"Warning: Model embedding dim {actual_dim} != expected {embedding_dim}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract text features.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            CLS embeddings [batch_size, embedding_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'architecture': 'BERT'
        }


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for images.
    Extracts CLS token representation from ViT.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        embedding_dim: int = 768,
        freeze: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ViT encoder.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: Output embedding dimension
            freeze: Whether to freeze pretrained weights
            device: Device to use
        """
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim

        print(f"Loading ViT model: {model_name}")
        self.vit = AutoModel.from_pretrained(model_name)

        if freeze:
            # Freeze all ViT parameters
            for param in self.vit.parameters():
                param.requires_grad = False

        # Verify embedding dimension
        actual_dim = self.vit.config.hidden_size
        if actual_dim != embedding_dim:
            print(f"Warning: Model embedding dim {actual_dim} != expected {embedding_dim}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.

        Args:
            pixel_values: Image pixels [batch_size, 3, height, width]

        Returns:
            CLS embeddings [batch_size, embedding_dim]
        """
        outputs = self.vit(pixel_values=pixel_values)

        # Use CLS token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'architecture': 'ViT'
        }


class CLIPEncoder(nn.Module):
    """
    CLIP encoder for image-text similarity measurement.
    Provides text and image embeddings for similarity calculation.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 512,
        freeze: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize CLIP encoder.

        Args:
            model_name: HuggingFace model identifier
            embedding_dim: CLIP embedding dimension
            freeze: Whether to freeze pretrained weights
            device: Device to use
        """
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim

        print(f"Loading CLIP model: {model_name}")
        self.clip = CLIPModel.from_pretrained(model_name)

        if freeze:
            # Freeze all CLIP parameters
            for param in self.clip.parameters():
                param.requires_grad = False

        # Set up processors
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode text to CLIP embedding.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Normalized text embeddings [batch_size, embedding_dim]
        """
        text_embeddings = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # L2 normalize
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)

        return text_embeddings

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image to CLIP embedding.

        Args:
            pixel_values: Image pixels [batch_size, 3, height, width]

        Returns:
            Normalized image embeddings [batch_size, embedding_dim]
        """
        image_embeddings = self.clip.get_image_features(pixel_values=pixel_values)

        # L2 normalize
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

        return image_embeddings

    def compute_similarity(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between text and image embeddings.

        Args:
            text_embeddings: [batch_size, embedding_dim]
            image_embeddings: [batch_size, embedding_dim]

        Returns:
            Cosine similarities [batch_size]
        """
        # Both are already L2 normalized, so dot product = cosine similarity
        similarities = (text_embeddings * image_embeddings).sum(dim=-1)
        return similarities

    def forward(
        self,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
        clip_pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode both modalities and compute similarity.

        Args:
            clip_input_ids: CLIP text token IDs
            clip_attention_mask: CLIP text attention mask
            clip_pixel_values: CLIP image pixels

        Returns:
            Tuple of (text_embeddings, image_embeddings, similarities)
        """
        text_emb = self.encode_text(clip_input_ids, clip_attention_mask)
        image_emb = self.encode_image(clip_pixel_values)
        similarity = self.compute_similarity(text_emb, image_emb)

        return text_emb, image_emb, similarity

    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'architecture': 'CLIP'
        }


class EncoderEnsemble(nn.Module):
    """
    Combined encoder that runs BERT, ViT, and CLIP together.
    Efficient for batch processing.
    """

    def __init__(
        self,
        config: Optional[EncoderConfig] = None,
        text_encoder_name: str = "google-bert/bert-base-uncased",
        image_encoder_name: str = "google/vit-base-patch16-224",
        clip_name: str = "openai/clip-vit-base-patch32",
        text_embedding_dim: int = 768,
        image_embedding_dim: int = 768,
        clip_embedding_dim: int = 512,
        freeze: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize encoder ensemble.

        Args:
            config: Optional EncoderConfig object
            text_encoder_name: BERT model name
            image_encoder_name: ViT model name
            clip_name: CLIP model name
            text_embedding_dim: BERT output dimension
            image_embedding_dim: ViT output dimension
            clip_embedding_dim: CLIP embedding dimension
            freeze: Freeze pretrained weights
            device: Device to use
        """
        super().__init__()

        if config is not None:
            self.text_encoder = BERTEncoder(
                model_name=config.text_encoder_name,
                embedding_dim=config.text_embedding_dim,
                freeze=config.freeze,
                device=config.device
            )
            self.image_encoder = ViTEncoder(
                model_name=config.image_encoder_name,
                embedding_dim=config.image_embedding_dim,
                freeze=config.freeze,
                device=config.device
            )
            self.clip_encoder = CLIPEncoder(
                model_name=config.clip_name,
                embedding_dim=config.clip_embedding_dim,
                freeze=config.freeze,
                device=config.device
            )
        else:
            self.text_encoder = BERTEncoder(
                model_name=text_encoder_name,
                embedding_dim=text_embedding_dim,
                freeze=freeze,
                device=device
            )
            self.image_encoder = ViTEncoder(
                model_name=image_encoder_name,
                embedding_dim=image_embedding_dim,
                freeze=freeze,
                device=device
            )
            self.clip_encoder = CLIPEncoder(
                model_name=clip_name,
                embedding_dim=clip_embedding_dim,
                freeze=freeze,
                device=device
            )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
        clip_pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from all encoders.

        Args:
            text_input_ids: BERT text tokens
            text_attention_mask: BERT attention mask
            pixel_values: ViT image pixels
            clip_input_ids: CLIP text tokens
            clip_attention_mask: CLIP text attention mask
            clip_pixel_values: CLIP image pixels

        Returns:
            Dictionary with text_features, image_features, clip_similarity
        """
        # BERT text features
        text_features = self.text_encoder(text_input_ids, text_attention_mask)

        # ViT image features
        image_features = self.image_encoder(pixel_values)

        # CLIP similarity
        _, _, clip_similarity = self.clip_encoder(
            clip_input_ids,
            clip_attention_mask,
            clip_pixel_values
        )

        return {
            'text_features': text_features,
            'image_features': image_features,
            'clip_similarity': clip_similarity
        }

    def extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract text features only."""
        return self.text_encoder(input_ids, attention_mask)

    def extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features only."""
        return self.image_encoder(pixel_values)

    def compute_clip_similarity(
        self,
        clip_input_ids: torch.Tensor,
        clip_attention_mask: torch.Tensor,
        clip_pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute CLIP similarity only."""
        _, _, similarity = self.clip_encoder(
            clip_input_ids,
            clip_attention_mask,
            clip_pixel_values
        )
        return similarity
