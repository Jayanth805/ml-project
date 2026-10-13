"""
Preprocessing utilities for text and image data.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    CLIPProcessor,
    CLIPModel,
    AutoModel,
    BertModel,
    ViTModel
)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""
    max_text_length: int = 128
    image_size: Tuple[int, int] = (224, 224)
    device: str = "cuda"


class TextPreprocessor:
    """
    Text preprocessing using BERT tokenizer.
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        max_length: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Initialize text preprocessor.

        Args:
            model_name: Name of the BERT model
            max_length: Maximum sequence length in tokens
            device: Device to use
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Dictionary with input_ids, attention_mask tensors
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        return {k: v.to(self.device) for k, v in encoded.items()}

    def decode(self, input_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text."""
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)


class ImagePreprocessor:
    """
    Image preprocessing for Vision Transformer.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[torch.device] = None
    ):
        """
        Initialize image preprocessor.

        Args:
            model_name: Name of the ViT model
            device: Device to use
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading image processor: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess image(s).

        Args:
            images: Single PIL Image or list of images

        Returns:
            Dictionary with pixel_values tensor
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Ensure RGB
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        encoded = self.processor(
            images=images,
            return_tensors='pt'
        )

        return {k: v.to(self.device) for k, v in encoded.items()}


class CLIPPreprocessor:
    """
    Preprocessing for CLIP model.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None
    ):
        """
        Initialize CLIP preprocessor.

        Args:
            model_name: Name of the CLIP model
            device: Device to use
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading CLIP processor: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None
    ) -> Dict[str, Any]:
        """
        Preprocess text and/or images for CLIP.

        Args:
            texts: Single text or list of texts
            images: Single PIL Image or list of images

        Returns:
            Dictionary with tokenized inputs
        """
        return self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

    def process_text(
        self,
        texts: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Process text only."""
        if isinstance(texts, str):
            texts = [texts]
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

    def process_images(
        self,
        images: Union[Image.Image, List[Image.Image]]
    ) -> Dict[str, torch.Tensor]:
        """Process images only."""
        if isinstance(images, Image.Image):
            images = [images]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
        return self.processor(
            images=images,
            return_tensors="pt"
        )


class MultimodalPreprocessor:
    """
    Combined preprocessing for text and images.
    """

    def __init__(
        self,
        text_model_name: str = "google-bert/bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        max_text_length: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Initialize multimodal preprocessor.

        Args:
            text_model_name: BERT model name
            image_model_name: ViT model name
            clip_model_name: CLIP model name
            max_text_length: Maximum text length in tokens
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_preprocessor = TextPreprocessor(
            model_name=text_model_name,
            max_length=max_text_length,
            device=self.device
        )

        self.image_preprocessor = ImagePreprocessor(
            model_name=image_model_name,
            device=self.device
        )

        self.clip_preprocessor = CLIPPreprocessor(
            model_name=clip_model_name,
            device=self.device
        )

        self.max_text_length = max_text_length

    def preprocess_sample(
        self,
        text: str,
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Preprocess a single sample.

        Args:
            text: Input text
            image: Optional input image

        Returns:
            Dictionary with preprocessed tensors
        """
        result = {}

        # Preprocess text
        text_inputs = self.text_preprocessor(text)
        result['text_input_ids'] = text_inputs['input_ids']
        result['text_attention_mask'] = text_inputs['attention_mask']

        # Preprocess image if available
        if image is not None:
            image_inputs = self.image_preprocessor(image)
            result['pixel_values'] = image_inputs['pixel_values']

            # CLIP image encoding
            clip_image_inputs = self.clip_preprocessor.process_images(image)
            result['clip_pixel_values'] = clip_image_inputs['pixel_values']

            # CLIP text encoding
            clip_text_inputs = self.clip_preprocessor.process_text(text)
            result['clip_input_ids'] = clip_text_inputs['input_ids']
            result['clip_attention_mask'] = clip_text_inputs['attention_mask']

        return result

    def preprocess_batch(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Preprocess a batch of samples.

        Args:
            texts: List of input texts
            images: Optional list of input images

        Returns:
            Dictionary with preprocessed tensors
        """
        result = {}

        # Preprocess texts
        text_inputs = self.text_preprocessor(texts)
        result['text_input_ids'] = text_inputs['input_ids']
        result['text_attention_mask'] = text_inputs['attention_mask']

        # Preprocess images if available
        if images is not None and any(img is not None for img in images):
            # Handle missing images by replacing with black image
            processed_images = []
            for img in images:
                if img is None:
                    # Create black image of same size
                    img = Image.new('RGB', (224, 224), color=(0, 0, 0))
                processed_images.append(img)

            image_inputs = self.image_preprocessor(processed_images)
            result['pixel_values'] = image_inputs['pixel_values']

            # CLIP encoding
            clip_image_inputs = self.clip_preprocessor.process_images(processed_images)
            result['clip_pixel_values'] = clip_image_inputs['pixel_values']

            clip_text_inputs = self.clip_preprocessor.process_text(texts)
            result['clip_input_ids'] = clip_text_inputs['input_ids']
            result['clip_attention_mask'] = clip_text_inputs['attention_mask']

        return result


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load an image from path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image or None if loading fails
    """
    try:
        img = Image.open(image_path)
        return img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def create_black_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a black placeholder image."""
    return Image.new('RGB', size, color=(0, 0, 0))
