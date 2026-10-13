"""
Utility functions for the multimodal fake news detection project.
"""

import os
import json
import yaml
import random
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation."""
    if preferred_device == "cpu":
        return torch.device("cpu")

    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    print("No GPU available, using CPU")
    return torch.device("cpu")


def print_gpu_info() -> Dict[str, Any]:
    """Print and return GPU information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {info['gpu_memory']:.2f} GB")

    return info


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_file_hash(filepath: Union[str, Path]) -> str:
    """Get MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def timestamp_now() -> str:
    """Get current timestamp as ISO format string."""
    return datetime.datetime.now().isoformat()


def format_timestamp(dt: Optional[datetime.datetime] = None) -> str:
    """Format datetime as a readable string."""
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clip_similarity_to_consistency_score(similarity: float) -> Dict[str, Any]:
    """
    Convert CLIP cosine similarity to a human-readable consistency interpretation.

    Note: This measures semantic alignment, NOT factual accuracy.
    A high score means the image semantically matches the text,
    but does NOT prove the news content is true.
    """
    if similarity >= 0.8:
        interpretation = "Strong semantic match"
        level = "high"
    elif similarity >= 0.6:
        interpretation = "Moderate semantic match"
        level = "medium-high"
    elif similarity >= 0.4:
        interpretation = "Weak semantic match"
        level = "medium"
    elif similarity >= 0.2:
        interpretation = "Low semantic alignment"
        level = "low"
    else:
        interpretation = "No semantic alignment"
        level = "very-low"

    return {
        "raw_similarity": float(similarity),
        "interpretation": interpretation,
        "level": level,
        "warning": "Semantic similarity does not indicate factual accuracy"
    }


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.1f}%"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_class_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Get class distribution from labels array."""
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(unique, counts)}


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    n_classes = len(unique)

    weights = {}
    for u, c in zip(unique, counts):
        # Inverse frequency weighting
        weights[int(u)] = total / (n_classes * c)

    return weights


class Configurable:
    """Base class for objects that can be initialized from config."""

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs):
        """Create instance from configuration dictionary."""
        return cls(**config, **kwargs)

    def to_config(self) -> Dict[str, Any]:
        """Convert instance to configuration dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: torch.nn.Module, model_name: str = "Model") -> None:
    """Print a summary of model parameters."""
    total, trainable = count_parameters(model)
    print(f"\n{model_name} Summary:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {total - trainable:,}")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """Find files matching pattern in directory."""
    return list(Path(directory).glob(pattern))
