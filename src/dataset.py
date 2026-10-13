"""
Dataset loading and preprocessing for multimodal fake news detection.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from .utils import ensure_dir, load_config


@dataclass
class Sample:
    """Represents a single sample in the dataset."""
    sample_id: str
    text: str
    image_path: Optional[str]
    label: int
    metadata: Dict[str, Any]


class MultimodalFakeNewsDataset(Dataset):
    """
    Dataset for multimodal fake news detection.

    Handles loading from Fakeddit format and provides text, image, and label data.
    """

    def __init__(
        self,
        samples: List[Sample],
        transform: Optional[Callable] = None,
        cache_images: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            samples: List of Sample objects
            transform: Optional transform for images
            cache_images: Whether to cache loaded images in memory
        """
        self.samples = samples
        self.transform = transform
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]

        item = {
            'sample_id': sample.sample_id,
            'text': sample.text,
            'label': sample.label
        }

        # Load image if path exists
        if sample.image_path and os.path.exists(sample.image_path):
            try:
                image = Image.open(sample.image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                item['image'] = image
                item['image_available'] = True
            except Exception as e:
                item['image'] = None
                item['image_available'] = False
                item['image_error'] = str(e)
        else:
            item['image'] = None
            item['image_available'] = False

        return item

    @classmethod
    def from_fakeddit(
        cls,
        csv_path: str,
        images_dir: str,
        text_column: str = 'clean_title',
        image_column: str = 'image_id',
        label_column: str = 'label',
        label_mapping: Optional[Dict[int, int]] = None,
        max_samples: Optional[int] = None,
        min_text_length: int = 5
    ) -> 'MultimodalFakeNewsDataset':
        """
        Load dataset from Fakeddit format.

        Args:
            csv_path: Path to the CSV file containing the dataset
            images_dir: Directory containing the images
            text_column: Name of the text column in CSV
            image_column: Name of the image ID column in CSV
            label_column: Name of the label column in CSV
            label_mapping: Optional mapping for binary classification
            max_samples: Maximum number of samples to load
            min_text_length: Minimum text length for valid samples

        Returns:
            MultimodalFakeNewsDataset instance
        """
        print(f"Loading Fakeddit dataset from {csv_path}...")

        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from CSV")

        # Apply label mapping for binary classification
        if label_mapping:
            df[label_column] = df[label_column].map(label_mapping)
            # Remove samples with unmapped labels
            df = df.dropna(subset=[label_column])
            df[label_column] = df[label_column].astype(int)

        # Filter valid samples
        df = df[df[text_column].notna()]
        df = df[df[text_column].str.len() >= min_text_length]

        # Add image path
        df['image_path'] = df[image_column].apply(
            lambda x: os.path.join(images_dir, f"{x}.jpg") if pd.notna(x) else None
        )

        # Check image availability (optional, can be slow for large datasets)
        # For now, assume all images exist
        df['image_exists'] = True

        # Limit samples if requested
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)

        # Create Sample objects
        samples = []
        for idx, row in df.iterrows():
            sample = Sample(
                sample_id=str(row.get('id', idx)),
                text=str(row[text_column]),
                image_path=row['image_path'] if row.get('image_exists', True) else None,
                label=int(row[label_column]),
                metadata={
                    'original_label': row.get('original_label', row[label_column]),
                    'subreddit': row.get('subreddit', 'unknown'),
                    'created_utc': row.get('created_utc', None)
                }
            )
            samples.append(sample)

        print(f"Created dataset with {len(samples)} samples")

        # Print class distribution
        labels = [s.label for s in samples]
        unique, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(labels)*100:.1f}%)")

        return cls(samples=samples)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        text_column: str,
        image_column: Optional[str] = None,
        label_column: Optional[str] = None,
        images_base_dir: Optional[str] = None,
        label_mapping: Optional[Dict[Any, int]] = None,
        max_samples: Optional[int] = None
    ) -> 'MultimodalFakeNewsDataset':
        """
        Load dataset from a generic CSV file.

        Args:
            csv_path: Path to the CSV file
            text_column: Name of the text column
            image_column: Optional name of image column (relative path or ID)
            label_column: Optional name of label column
            images_base_dir: Base directory for images if image_column contains relative paths
            label_mapping: Optional mapping for labels
            max_samples: Maximum number of samples

        Returns:
            MultimodalFakeNewsDataset instance
        """
        print(f"Loading dataset from {csv_path}...")

        df = pd.read_csv(csv_path)

        # Apply label mapping
        if label_mapping and label_column:
            df[label_column] = df[label_column].map(label_mapping)
            df = df.dropna(subset=[label_column])
            df[label_column] = df[label_column].astype(int)

        # Limit samples
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)

        # Create samples
        samples = []
        for idx, row in df.iterrows():
            # Handle image path
            if image_column and images_base_dir:
                image_id = row.get(image_column)
                if pd.notna(image_id):
                    image_path = os.path.join(images_base_dir, str(image_id))
                else:
                    image_path = None
            else:
                image_path = None

            # Handle label
            if label_column:
                label = int(row[label_column])
            else:
                label = -1  # Unknown label

            sample = Sample(
                sample_id=str(row.get('id', idx)),
                text=str(row[text_column]),
                image_path=image_path,
                label=label,
                metadata={}
            )
            samples.append(sample)

        print(f"Created dataset with {len(samples)} samples")
        return cls(samples=samples)

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        labels = [s.label for s in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def filter_by_image_availability(self) -> 'MultimodalFakeNewsDataset':
        """Return a new dataset with only samples that have valid images."""
        filtered_samples = []
        for sample in self.samples:
            if sample.image_path and os.path.exists(sample.image_path):
                filtered_samples.append(sample)
        print(f"Filtered to {len(filtered_samples)} samples with valid images")
        return MultimodalFakeNewsDataset(
            samples=filtered_samples,
            transform=self.transform,
            cache_images=self.cache_images
        )


class FeatureCache:
    """
    Cache for extracted features to avoid repeated computation.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = ensure_dir(cache_dir)

    def get_cache_path(self, model_name: str, split: str) -> Path:
        """Get the path for a cached feature file."""
        return self.cache_dir / f"{model_name}_{split}.pt"

    def save_features(
        self,
        features: torch.Tensor,
        model_name: str,
        split: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save features to cache."""
        cache_path = self.get_cache_path(model_name, split)
        data = {
            'features': features,
            'metadata': metadata or {}
        }
        torch.save(data, cache_path)
        print(f"Saved features to {cache_path}")

    def load_features(
        self,
        model_name: str,
        split: str
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Load features from cache."""
        cache_path = self.get_cache_path(model_name, split)
        if cache_path.exists():
            data = torch.load(cache_path)
            print(f"Loaded features from {cache_path}")
            return data['features'], data.get('metadata', {})
        return None

    def clear(self) -> None:
        """Clear all cached features."""
        for f in self.cache_dir.glob("*.pt"):
            f.unlink()
        print(f"Cleared feature cache at {self.cache_dir}")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.

    Handles variable-length sequences and missing images.
    """
    # This is handled by the custom dataset
    return batch


def create_data_loaders(
    train_dataset: MultimodalFakeNewsDataset,
    val_dataset: MultimodalFakeNewsDataset,
    test_dataset: MultimodalFakeNewsDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def stratified_split(
    dataset: MultimodalFakeNewsDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[MultimodalFakeNewsDataset, MultimodalFakeNewsDataset, MultimodalFakeNewsDataset]:
    """
    Split dataset into train, validation, and test sets with stratification.

    Args:
        dataset: The dataset to split
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    samples = dataset.samples
    labels = np.array([s.label for s in samples])

    # Get unique labels
    unique_labels = np.unique(labels)

    # Split indices by label
    train_indices = []
    val_indices = []
    test_indices = []

    np.random.seed(seed)

    for label in unique_labels:
        # Get indices for this label
        label_indices = np.where(labels == label)[0].tolist()
        np.random.shuffle(label_indices)

        n = len(label_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:])

    # Create new datasets
    train_dataset = MultimodalFakeNewsDataset(
        samples=[samples[i] for i in train_indices],
        transform=dataset.transform,
        cache_images=dataset.cache_images
    )

    val_dataset = MultimodalFakeNewsDataset(
        samples=[samples[i] for i in val_indices],
        transform=dataset.transform,
        cache_images=dataset.cache_images
    )

    test_dataset = MultimodalFakeNewsDataset(
        samples=[samples[i] for i in test_indices],
        transform=dataset.transform,
        cache_images=dataset.cache_images
    )

    print(f"Split dataset into:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset
