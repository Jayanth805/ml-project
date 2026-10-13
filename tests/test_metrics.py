"""
Tests for metrics module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_per_class_metrics,
    compare_models
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['macro_f1'] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0

    def test_mixed_predictions(self):
        """Test metrics with mixed predictions."""
        y_true = np.array([0, 1, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.8])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        # Should be 4 correct out of 6
        assert metrics['accuracy'] == pytest.approx(4/6, rel=1e-5)

    def test_sample_counts(self):
        """Test that sample counts are correct."""
        y_true = np.array([0, 1, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.6, 0.2, 0.8, 0.9])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics['n_samples'] == 6
        assert metrics['n_positive'] == 3
        assert metrics['n_negative'] == 3

    def test_without_probabilities(self):
        """Test metrics calculation without probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred, None)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'roc_auc' not in metrics  # Can't compute without probabilities


class TestConfusionMatrix:
    """Tests for compute_confusion_matrix function."""

    def test_binary_confusion_matrix(self):
        """Test binary confusion matrix."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])

        cm = compute_confusion_matrix(y_true, y_pred)

        # TN=2, FP=1, FN=1, TP=2
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 2  # TP

    def test_with_labels(self):
        """Test confusion matrix with custom labels."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        cm = compute_confusion_matrix(y_true, y_pred, labels=[0, 1])

        assert cm[0, 0] == 2
        assert cm[1, 1] == 2


class TestClassificationReport:
    """Tests for get_classification_report function."""

    def test_report_generation(self):
        """Test classification report generation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])

        report = get_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert 'precision' in report.lower()
        assert 'recall' in report.lower()
        assert 'f1-score' in report.lower()

    def test_custom_class_names(self):
        """Test report with custom class names."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        report = get_classification_report(y_true, y_pred, target_names=['REAL', 'FAKE'])

        assert 'REAL' in report
        assert 'FAKE' in report


class TestPerClassMetrics:
    """Tests for compute_per_class_metrics function."""

    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])

        per_class = compute_per_class_metrics(y_true, y_pred)

        assert 'REAL (0)' in per_class or 'Class_0' in per_class
        assert len(per_class) == 2


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_models(self):
        """Test model comparison."""
        results = {
            'model_a': {
                'accuracy': 0.85,
                'macro_f1': 0.84,
                'roc_auc': 0.90
            },
            'model_b': {
                'accuracy': 0.88,
                'macro_f1': 0.87,
                'roc_auc': 0.92
            },
            'model_c': {
                'accuracy': 0.82,
                'macro_f1': 0.81,
                'roc_auc': 0.88
            }
        }

        comparison = compare_models(results)

        assert 'models' in comparison
        assert comparison['models'] == ['model_a', 'model_b', 'model_c']
        assert comparison['best_by_accuracy'] == 'model_b'
        assert comparison['best_by_f1'] == 'model_b'
        assert comparison['best_by_auc'] == 'model_b'

    def test_ranking(self):
        """Test model ranking."""
        results = {
            'a': {'accuracy': 0.8},
            'b': {'accuracy': 0.9},
            'c': {'accuracy': 0.7}
        }

        comparison = compare_models(results)

        assert comparison['ranking']['accuracy'] == ['b', 'a', 'c']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
