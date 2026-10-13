"""
Plotting utilities for visualization of training and evaluation results.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .utils import ensure_dir, save_json


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    title: str = "Training History"
) -> Figure:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_losses' and 'val_losses'
        output_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history.get('train_losses', [])) + 1)

    if 'train_losses' in history:
        ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)

    if 'val_losses' in history:
        ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_validation_metrics(
    val_metrics: List[Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Validation Metrics"
) -> Figure:
    """
    Plot validation metrics over epochs.

    Args:
        val_metrics: List of metrics dicts, one per epoch
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(val_metrics) + 1)

    # Extract metrics
    metrics_to_plot = ['accuracy', 'macro_f1', 'precision', 'recall']
    colors = ['blue', 'green', 'orange', 'purple']

    for metric, color in zip(metrics_to_plot, colors):
        values = [m.get(metric, 0) for m in val_metrics]
        ax.plot(epochs, values, '-o', label=metric.replace('_', ' ').title(),
                color=color, linewidth=2, markersize=6)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        confusion_matrix: 2D numpy array
        class_names: Optional class labels
        output_path: Optional save path
        title: Plot title
        normalize: Whether to normalize percentages

    Returns:
        Matplotlib Figure
    """
    if class_names is None:
        class_names = ['REAL', 'FAKE']

    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    model_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Model Comparison"
) -> Figure:
    """
    Compare multiple models across metrics.

    Args:
        model_results: Dict mapping model name to metrics dict
        metrics: List of metrics to compare
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    if metrics is None:
        metrics = ['accuracy', 'macro_f1', 'precision', 'recall', 'roc_auc']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.8 / len(model_results)

    for i, (model_name, results) in enumerate(model_results.items()):
        values = [results.get(m, 0) for m in metrics]
        offset = (i - len(model_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curves(
    all_results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "ROC Curves"
) -> Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        all_results: Dict mapping model name to evaluation results
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, results in all_results.items():
        labels = np.array(results['labels'])
        probs = np.array(results['probabilities'])

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, linewidth=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_clip_similarity_distribution(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "CLIP Similarity Distribution"
) -> Figure:
    """
    Plot distribution of CLIP similarity scores.

    Args:
        results: Evaluation results with clip_similarities
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    clip_sims = np.array(results.get('clip_similarities', []))
    labels = np.array(results.get('labels', []))

    if len(clip_sims) > 0:
        # Separate by label
        real_sims = clip_sims[labels == 0]
        fake_sims = clip_sims[labels == 1]

        ax.hist(real_sims, bins=30, alpha=0.6, label='REAL', color='blue', density=True)
        ax.hist(fake_sims, bins=30, alpha=0.6, label='FAKE', color='red', density=True)

        # Add means
        ax.axvline(np.mean(real_sims), color='blue', linestyle='--', linewidth=2,
                   label=f'REAL mean: {np.mean(real_sims):.3f}')
        ax.axvline(np.mean(fake_sims), color='red', linestyle='--', linewidth=2,
                   label=f'FAKE mean: {np.mean(fake_sims):.3f}')

    ax.set_xlabel('CLIP Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Ablation Study"
) -> Figure:
    """
    Plot ablation study results.

    Args:
        ablation_results: Dict mapping variant name to metrics
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    variants = list(ablation_results.keys())
    metrics = ['accuracy', 'macro_f1']

    for ax, metric in zip(axes, metrics):
        values = [ablation_results[v].get(metric, 0) for v in variants]

        bars = ax.bar(variants, values, color=['blue', 'green', 'orange', 'purple'][:len(variants)])

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x labels
        ax.set_xticklabels(variants, rotation=45, ha='right')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_image_swap_comparison(
    original_results: Dict[str, float],
    swapped_results: Dict[str, float],
    output_path: Optional[str] = None,
    title: str = "Original vs Image-Swapped Performance"
) -> Figure:
    """
    Compare model performance on original vs swapped pairs.

    Args:
        original_results: Metrics for original pairs
        swapped_results: Metrics for swapped pairs
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['accuracy', 'precision', 'recall', 'macro_f1']
    x = np.arange(len(metrics))
    width = 0.35

    orig_values = [original_results.get(m, 0) for m in metrics]
    swap_values = [swapped_results.get(m, 0) for m in metrics]

    bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color='blue')
    bars2 = ax.bar(x + width/2, swap_values, width, label='Image-Swapped', color='red')

    for bar, val in zip(bars1, orig_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, swap_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Optional[str] = None,
    title: str = "Class Distribution"
) -> Figure:
    """
    Plot class distribution as bar chart and pie chart.

    Args:
        class_counts: Dict mapping class name to count
        output_path: Optional save path
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['blue', 'red'][:len(classes)]

    # Bar chart
    axes[0].bar(classes, counts, color=colors)
    for i, (cls, cnt) in enumerate(zip(classes, counts)):
        axes[0].text(i, cnt + max(counts) * 0.02, str(cnt),
                    ha='center', va='bottom', fontsize=11)
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Class Counts', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Pie chart
    axes[1].pie(counts, labels=classes, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Class Proportions', fontsize=12, fontweight='bold')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        ensure_dir(Path(output_path).parent)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def save_plots_summary(
    results: Dict[str, Any],
    output_dir: str,
    plots: Dict[str, str]
) -> None:
    """
    Save a summary of generated plots.

    Args:
        results: Results data
        output_dir: Output directory
        plots: Dict mapping plot name to file path
    """
    summary = {
        'generated_plots': plots,
        'results_summary': {
            'n_samples': results.get('metrics', {}).get('n_samples', 0),
            'accuracy': results.get('metrics', {}).get('accuracy', 0),
            'macro_f1': results.get('metrics', {}).get('macro_f1', 0)
        }
    }

    save_json(summary, Path(output_dir) / 'plots_summary.json')
