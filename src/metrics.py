"""
Evaluation metrics for fake news detection.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    positive_class: int = 1
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
        positive_class: Which class is considered positive (default: 1 = FAKE)

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 for positive class
    metrics['precision'] = precision_score(
        y_true, y_pred, pos_label=positive_class, zero_division=0
    )
    metrics['recall'] = recall_score(
        y_true, y_pred, pos_label=positive_class, zero_division=0
    )
    metrics['f1'] = f1_score(
        y_true, y_pred, pos_label=positive_class, zero_division=0
    )

    # Macro metrics (average across classes)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted metrics
    metrics['weighted_precision'] = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_recall'] = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_f1'] = f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    for cls in np.unique(y_true):
        metrics[f'class_{int(cls)}_precision'] = precision_score(
            y_true, y_pred, labels=[cls], average='micro', zero_division=0
        )
        metrics[f'class_{int(cls)}_recall'] = recall_score(
            y_true, y_pred, labels=[cls], average='micro', zero_division=0
        )
        metrics[f'class_{int(cls)}_f1'] = f1_score(
            y_true, y_pred, labels=[cls], average='micro', zero_division=0
        )

    # ROC-AUC (requires probabilities)
    if y_prob is not None:
        try:
            # Binary case
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            else:
                # Multi-class case
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
        except ValueError as e:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0

    # Sample counts
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true == positive_class))
    metrics['n_negative'] = int(np.sum(y_true != positive_class))

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional label ordering

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Get detailed classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Optional class names

    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ['REAL (0)', 'FAKE (1)']

    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Dictionary mapping class name to metrics dict
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in np.unique(y_true)]

    unique_classes = np.unique(y_true)
    per_class = {}

    for i, cls in enumerate(unique_classes):
        # Binary mask for this class
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        # True positives, false positives, false negatives
        tp = int(np.sum((y_true_cls == 1) & (y_pred_cls == 1)))
        fp = int(np.sum((y_true_cls == 0) & (y_pred_cls == 1)))
        fn = int(np.sum((y_true_cls == 1) & (y_pred_cls == 0)))
        tn = int(np.sum((y_true_cls == 0) & (y_pred_cls == 0)))

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[class_names[i] if i < len(class_names) else f"Class_{cls}"] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }

    return per_class


def compute_confidence_intervals(
    metrics: Dict[str, float],
    n_samples: int,
    confidence: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for metrics using binomial approximation.

    Note: This is an approximation. For precise intervals, use bootstrapping.

    Args:
        metrics: Dictionary of metric values
        n_samples: Number of samples
        confidence: Confidence level (default: 0.95)

    Returns:
        Dictionary of metric -> (lower, upper) tuple
    """
    from scipy import stats

    z_score = stats.norm.ppf((1 + confidence) / 2)

    intervals = {}
    for name, value in metrics.items():
        if name in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
            # Binomial standard error
            se = np.sqrt(value * (1 - value) / n_samples)
            margin = z_score * se
            intervals[name] = (max(0, value - margin), min(1, value + margin))
        elif name == 'roc_auc':
            # Approximate SE for AUC (DeLong method approximation)
            se = np.sqrt(value * (1 - value) / n_samples)
            margin = z_score * se
            intervals[name] = (max(0, value - margin), min(1, value + margin))

    return intervals


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth
        y_pred: Predictions
        y_prob: Probabilities (optional)
        metric_fn: Function that computes the metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (value, lower_bound, upper_bound)
    """
    n = len(y_true)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices] if y_prob is not None else None

        try:
            value = metric_fn(y_true_boot, y_pred_boot, y_prob_boot)
            bootstrap_values.append(value)
        except:
            pass

    if not bootstrap_values:
        return 0.0, 0.0, 0.0

    bootstrap_values = np.array(bootstrap_values)
    value = metric_fn(y_true, y_pred, y_prob)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return value, lower, upper


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Optional title for the output
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")

    print(f"{'='*50}\n")


def compare_models(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Compare multiple model results.

    Args:
        results: Dictionary mapping model name to metrics dict

    Returns:
        Comparison summary dictionary
    """
    comparison = {
        'models': list(results.keys()),
        'best_by_accuracy': max(results.keys(), key=lambda k: results[k].get('accuracy', 0)),
        'best_by_f1': max(results.keys(), key=lambda k: results[k].get('macro_f1', 0)),
        'best_by_auc': max(results.keys(), key=lambda k: results[k].get('roc_auc', 0)),
        'ranking': {}
    }

    # Rank by each metric
    for metric in ['accuracy', 'macro_f1', 'roc_auc', 'precision', 'recall']:
        sorted_models = sorted(
            results.keys(),
            key=lambda k: results[k].get(metric, 0),
            reverse=True
        )
        comparison['ranking'][metric] = sorted_models

    return comparison
