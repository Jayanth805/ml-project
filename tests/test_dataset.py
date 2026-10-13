"""
Tests for dataset module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    MultimodalFakeNewsDataset,
    Sample,
    FeatureCache,
    stratified_split
)


class TestSample:
    """Tests for Sample dataclass."""

    def test_sample_creation(self):
        """Test creating a sample."""
        sample = Sample(
            sample_id="test_1",
            text="Test news headline",
            image_path="/path/to/image.jpg",
            label=1,
            metadata={"source": "test"}
        )

        assert sample.sample_id == "test_1"
        assert sample.text == "Test news headline"
        assert sample.image_path == "/path/to/image.jpg"
        assert sample.label == 1
        assert sample.metadata["source"] == "test"

    def test_sample_optional_image(self):
        """Test sample without image."""
        sample = Sample(
            sample_id="test_2",
            text="Text only",
            image_path=None,
            label=0,
            metadata={}
        )

        assert sample.image_path is None


class TestMultimodalFakeNewsDataset:
    """Tests for MultimodalFakeNewsDataset."""

    def test_dataset_creation(self):
        """Test creating a dataset from samples."""
        samples = [
            Sample("1", "Text 1", None, 0, {}),
            Sample("2", "Text 2", None, 1, {}),
            Sample("3", "Text 3", None, 0, {}),
        ]

        dataset = MultimodalFakeNewsDataset(samples)

        assert len(dataset) == 3

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        samples = [
            Sample("1", "First text", None, 0, {}),
            Sample("2", "Second text", None, 1, {}),
        ]

        dataset = MultimodalFakeNewsDataset(samples)
        item = dataset[0]

        assert item['sample_id'] == "1"
        assert item['text'] == "First text"
        assert item['label'] == 0

    def test_class_distribution(self):
        """Test getting class distribution."""
        samples = [
            Sample("1", "Text 1", None, 0, {}),
            Sample("2", "Text 2", None, 1, {}),
            Sample("3", "Text 3", None, 1, {}),
        ]

        dataset = MultimodalFakeNewsDataset(samples)
        dist = dataset.get_class_distribution()

        assert dist[0] == 1
        assert dist[1] == 2


class TestStratifiedSplit:
    """Tests for stratified split function."""

    def test_stratified_split_ratios(self):
        """Test that split produces correct ratios."""
        samples = [
            Sample(str(i), f"Text {i}", None, i % 2, {})
            for i in range(100)
        ]

        dataset = MultimodalFakeNewsDataset(samples)
        train, val, test = stratified_split(dataset, 0.7, 0.15, 0.15, seed=42)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_stratified_split_preserves_labels(self):
        """Test that splits preserve class distribution."""
        samples = [
            Sample(str(i), f"Text {i}", None, i % 2, {})
            for i in range(100)
        ]

        dataset = MultimodalFakeNewsDataset(samples)
        train, val, test = stratified_split(dataset, 0.7, 0.15, 0.15, seed=42)

        # Check each split has both classes
        for split_dataset in [train, val, test]:
            labels = [s.label for s in split_dataset.samples]
            assert 0 in labels
            assert 1 in labels


class TestFeatureCache:
    """Tests for FeatureCache."""

    def test_cache_path_generation(self, tmp_path):
        """Test cache path generation."""
        cache = FeatureCache(str(tmp_path))

        path = cache.get_cache_path("bert", "train")
        assert path.name == "bert_train.pt"

    def test_cache_roundtrip(self, tmp_path):
        """Test saving and loading features from cache."""
        import torch

        cache = FeatureCache(str(tmp_path))

        # Create dummy features
        features = torch.randn(10, 768)
        labels = torch.randint(0, 2, (10,))

        # Save
        cache.save_features(features, "test", "train", {"n_samples": 10})

        # Load
        loaded_features, metadata = cache.load_features("test", "train")

        assert torch.allclose(features, loaded_features)
        assert metadata["n_samples"] == 10

    def test_cache_miss(self, tmp_path):
        """Test loading non-existent cache."""
        cache = FeatureCache(str(tmp_path))

        result = cache.load_features("nonexistent", "train")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
