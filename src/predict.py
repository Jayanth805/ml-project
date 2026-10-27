"""
Inference pipeline for news analysis.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

import torch
from PIL import Image

from .classifier import MultimodalClassifier, ClassifierConfig
from .preprocessing import MultimodalPreprocessor, load_image
from .utils import (
    get_device, clip_similarity_to_consistency_score,
    format_probability, timestamp_now, ensure_dir
)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    prediction: str  # "FAKE" or "REAL"
    probability_fake: float
    probability_real: float
    confidence: float
    clip_similarity: float
    consistency_interpretation: Dict[str, Any]
    model_type: str
    model_version: str
    timestamp: str
    text_preview: str
    image_filename: Optional[str]
    disclaimer: str = (
        "This prediction is based on learned patterns and semantic consistency. "
        "It does not verify factual accuracy. Semantic similarity between image "
        "and text does not prove the news content is true or false."
    )


class NewsAnalyzer:
    """
    Main interface for analyzing news articles and images.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[ClassifierConfig] = None,
        device: Optional[torch.device] = None,
        model_type: str = "consistency_aware"
    ):
        """
        Initialize the news analyzer.

        Args:
            checkpoint_path: Path to model checkpoint (optional)
            config: Model configuration (optional)
            device: Device to use
            model_type: Type of model to use if no checkpoint provided
        """
        self.device = device or get_device()
        self.model_type = model_type

        # Initialize preprocessor
        self.preprocessor = MultimodalPreprocessor(device=self.device)

        # Initialize model
        if config is None:
            config = ClassifierConfig(model_type=model_type)

        self.model = MultimodalClassifier(config)
        self.model.to(self.device)
        self.model.eval()

        # Load checkpoint if provided
        self.checkpoint_loaded = False
        self.checkpoint_info = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load a model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            if 'model_state_dict' in checkpoint:
                saved_model_config = checkpoint.get('config', {}).get('model_config')
                if saved_model_config:
                    model_config = ClassifierConfig(**saved_model_config)
                    if model_config != self.model.config:
                        self.model = MultimodalClassifier(model_config).to(self.device)
                        self.model.eval()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.checkpoint_info = checkpoint.get('config', {})
                self.model_type = self.checkpoint_info.get('model_config', {}).get(
                    'model_type', self.model_type
                )
            else:
                self.model.load_state_dict(checkpoint)

            self.checkpoint_loaded = True
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            self.checkpoint_loaded = False
            return False

    def analyze(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        threshold: float = 0.5
    ) -> PredictionResult:
        """
        Analyze a news article and/or image.

        Args:
            text: News headline or article text
            image: Path to image file or PIL Image (optional)
            threshold: Classification threshold

        Returns:
            PredictionResult with analysis output
        """
        text = text or ""
        if self.model_type == "image_only":
            if image is None:
                raise ValueError("The image-only model requires an image")
            if not text.strip():
                text = "Image-only classification"
        elif not text.strip():
            raise ValueError("Text input cannot be empty")
        elif self.model_type != "text_only" and image is None:
            raise ValueError(f"The {self.model_type} model requires both text and an image")

        # Load image if path provided
        image_obj = None
        image_filename = None

        if image is not None:
            if isinstance(image, str):
                image_obj = load_image(image)
                image_filename = os.path.basename(image)
                if image_obj is None:
                    print(f"Warning: Could not load image from {image}")
            else:
                image_obj = image
                image_filename = "uploaded_image"

        # Preprocess
        inputs = self.preprocessor.preprocess_sample(text, image_obj)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get CLIP similarity for interpretation (even if no checkpoint)
        clip_sim = None
        if 'clip_pixel_values' in inputs and 'clip_input_ids' in inputs:
            with torch.no_grad():
                clip_similarity = self.model.encoders.compute_clip_similarity(
                    inputs['clip_input_ids'],
                    inputs['clip_attention_mask'],
                    inputs['clip_pixel_values']
                )
                clip_sim = clip_similarity.item()

        # Get prediction (only if checkpoint loaded)
        if self.checkpoint_loaded:
            with torch.no_grad():
                logits, intermediate = self.model(**inputs)
                prob_fake = torch.sigmoid(logits.squeeze(-1)).item()

                # Update CLIP similarity from model output if available
                intermediate_clip_similarity = intermediate.get('clip_similarity')
                if intermediate_clip_similarity is not None:
                    clip_sim = intermediate_clip_similarity.item()
        else:
            # No checkpoint - return a placeholder result
            prob_fake = 0.5
            clip_sim = clip_sim if clip_sim is not None else 0.0

        prob_real = 1 - prob_fake

        # Determine prediction
        prediction = "FAKE" if prob_fake >= threshold else "REAL"

        # Compute confidence
        confidence = max(prob_fake, prob_real)

        # CLIP similarity interpretation
        consistency_interpretation = clip_similarity_to_consistency_score(clip_sim) if clip_sim else {
            "raw_similarity": None,
            "interpretation": "No image provided",
            "level": "unknown",
            "warning": "Image required for consistency analysis"
        }

        # Create result
        result = PredictionResult(
            prediction=prediction,
            probability_fake=prob_fake,
            probability_real=prob_real,
            confidence=confidence,
            clip_similarity=clip_sim if clip_sim else 0.0,
            consistency_interpretation=consistency_interpretation,
            model_type=self.model_type,
            model_version=self.checkpoint_info.get('timestamp', 'untrained') if self.checkpoint_info else 'untrained',
            timestamp=timestamp_now(),
            text_preview=text[:200] + "..." if len(text) > 200 else text,
            image_filename=image_filename
        )

        return result

    def analyze_batch(
        self,
        samples: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> List[PredictionResult]:
        """
        Analyze a batch of samples.

        Args:
            samples: List of dicts with 'text' and optional 'image' keys
            threshold: Classification threshold

        Returns:
            List of PredictionResult objects
        """
        results = []

        for sample in samples:
            text = sample.get('text', '')
            image = sample.get('image')
            result = self.analyze(text, image, threshold)
            results.append(result)

        return results

    def is_ready(self) -> bool:
        """Check if a trained model checkpoint is loaded."""
        return self.checkpoint_loaded


class EnsembleAnalyzer:
    """
    Ensemble of multiple NewsAnalyzers for more robust predictions.
    """

    def __init__(self, analyzers: List[NewsAnalyzer]):
        """
        Initialize ensemble.

        Args:
            analyzers: List of NewsAnalyzer instances
        """
        self.analyzers = analyzers

    def analyze(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyze using ensemble voting.

        Args:
            text: News text
            image: Optional image
            threshold: Classification threshold

        Returns:
            Dictionary with ensemble results
        """
        individual_results = []

        for analyzer in self.analyzers:
            result = analyzer.analyze(text, image, threshold)
            individual_results.append(result)

        # Average probabilities
        avg_prob_fake = sum(r.probability_fake for r in individual_results) / len(individual_results)
        avg_prob_real = 1 - avg_prob_fake
        avg_clip_sim = sum(r.clip_similarity for r in individual_results) / len(individual_results)

        # Voting
        predictions = [r.prediction for r in individual_results]
        vote_fake = predictions.count("FAKE")
        vote_real = predictions.count("REAL")
        ensemble_prediction = "FAKE" if vote_fake > vote_real else "REAL"

        # Confidence based on agreement
        agreement = max(vote_fake, vote_real) / len(analyzers)

        return {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_probability_fake': avg_prob_fake,
            'ensemble_probability_real': avg_prob_real,
            'ensemble_clip_similarity': avg_clip_sim,
            'voting_agreement': agreement,
            'individual_results': individual_results
        }
