"""
Evaluation module for model evaluation on test sets.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .classifier import MultimodalClassifier
from .preprocessing import MultimodalPreprocessor
from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_per_class_metrics,
    print_metrics,
    compare_models
)
from .utils import ensure_dir, save_json, timestamp_now


class Evaluator:
    """
    Comprehensive model evaluator.
    """

    def __init__(
        self,
        model: MultimodalClassifier,
        preprocessor: MultimodalPreprocessor,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            preprocessor: Multimodal preprocessor
            device: Device to use
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        data_loader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: Data loader for evaluation
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation results
        """
        all_preds = []
        all_labels = []
        all_probs = []
        all_clip_similarities = []
        all_sample_ids = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Prepare inputs
                texts = batch['text']
                images = [b if b is not None else None for b in batch.get('image', [None]*len(batch['text']))]
                labels = batch['label'].to(device=self.device, dtype=torch.float32)

                # Preprocess
                inputs = self.preprocessor.preprocess_batch(texts, images)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                logits, intermediate = self.model(**inputs)

                # Get predictions
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                preds = (probs >= threshold).astype(int)

                # Store results
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                all_clip_similarities.extend(
                    intermediate.get('clip_similarity', torch.zeros(len(labels))).cpu().numpy()
                )
                all_sample_ids.extend(batch.get('sample_id', [f"sample_{i}" for i in range(len(labels))]))

        # Convert to arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_clip_similarities = np.array(all_clip_similarities)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        confusion = compute_confusion_matrix(all_labels, all_preds)
        per_class = compute_per_class_metrics(all_labels, all_preds)

        # Build results
        results = {
            'metrics': metrics,
            'confusion_matrix': confusion.tolist(),
            'per_class_metrics': per_class,
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist(),
            'labels': all_labels.tolist(),
            'clip_similarities': all_clip_similarities.tolist(),
            'sample_ids': all_sample_ids,
            'threshold': threshold,
            'timestamp': timestamp_now()
        }

        return results

    def evaluate_image_swap(
        self,
        data_loader: DataLoader,
        swapped_data_loader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on original and image-swapped data.

        Args:
            data_loader: Original data loader
            swapped_data_loader: Data with swapped images
            threshold: Classification threshold

        Returns:
            Dictionary with comparison results
        """
        print("Evaluating on original pairs...")
        original_results = self.evaluate(data_loader, threshold)

        print("Evaluating on swapped pairs...")
        swapped_results = self.evaluate(swapped_data_loader, threshold)

        # Compute comparison metrics
        comparison = {
            'original': original_results,
            'swapped': swapped_results,
            'metric_differences': {}
        }

        # Compute differences
        for metric in ['accuracy', 'macro_f1', 'precision', 'recall', 'roc_auc']:
            orig_val = original_results['metrics'].get(metric, 0)
            swap_val = swapped_results['metrics'].get(metric, 0)
            comparison['metric_differences'][metric] = swap_val - orig_val

        # Analyze prediction changes
        orig_preds = np.array(original_results['predictions'])
        swap_preds = np.array(swapped_results['predictions'])
        orig_labels = np.array(original_results['labels'])

        # Find samples that changed prediction
        changed_mask = orig_preds != swap_preds
        comparison['n_changed_predictions'] = int(np.sum(changed_mask))
        comparison['change_rate'] = float(np.mean(changed_mask))

        # Among changed, how many became fake/real
        orig_fake = (orig_preds == 1)
        swap_fake = (swap_preds == 1)
        became_fake = np.sum(~orig_fake & swap_fake)
        became_real = np.sum(orig_fake & ~swap_fake)
        comparison['became_fake'] = int(became_fake)
        comparison['became_real'] = int(became_real)

        return comparison

    def generate_plots(
        self,
        results: Dict[str, Any],
        output_dir: str,
        model_name: str = "model"
    ) -> Dict[str, str]:
        """
        Generate evaluation plots.

        Args:
            results: Evaluation results
            output_dir: Directory to save plots
            model_name: Name for saved files

        Returns:
            Dictionary mapping plot name to file path
        """
        output_dir = ensure_dir(output_dir)
        plot_paths = {}

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['REAL', 'FAKE'],
                    yticklabels=['REAL', 'FAKE'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        path = str(output_dir / f'{model_name}_confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['confusion_matrix'] = path

        # Probability distribution
        plt.figure(figsize=(10, 6))
        probs = np.array(results['probabilities'])
        labels = np.array(results['labels'])

        plt.hist(probs[labels == 0], bins=30, alpha=0.6, label='REAL', color='blue')
        plt.hist(probs[labels == 1], bins=30, alpha=0.6, label='FAKE', color='red')
        plt.xlabel('Probability of FAKE')
        plt.ylabel('Count')
        plt.title(f'Prediction Probability Distribution - {model_name}')
        plt.legend()
        path = str(output_dir / f'{model_name}_prob_dist.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['probability_distribution'] = path

        # CLIP similarity distribution
        plt.figure(figsize=(10, 6))
        clip_sims = np.array(results['clip_similarities'])
        plt.hist(clip_sims, bins=30, alpha=0.7, color='green')
        plt.xlabel('CLIP Similarity Score')
        plt.ylabel('Count')
        plt.title(f'CLIP Image-Text Similarity Distribution - {model_name}')
        path = str(output_dir / f'{model_name}_clip_similarity.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['clip_similarity'] = path

        # ROC curve
        if 'roc_auc' in results['metrics']:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(labels, probs)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, 'b-', linewidth=2,
                     label=f'ROC (AUC = {results["metrics"]["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            path = str(output_dir / f'{model_name}_roc_curve.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths['roc_curve'] = path

        return plot_paths


def evaluate_models(
    models: Dict[str, MultimodalClassifier],
    preprocessor: MultimodalPreprocessor,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple models on the same test set.

    Args:
        models: Dictionary mapping model name to model
        preprocessor: Multimodal preprocessor
        test_loader: Test data loader
        device: Device to use
        output_dir: Optional directory for results

    Returns:
        Dictionary mapping model name to results
    """
    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        evaluator = Evaluator(model, preprocessor, device)
        results = evaluator.evaluate(test_loader)

        print_metrics(results['metrics'], f"{model_name} Results")
        all_results[model_name] = results

        # Save results
        if output_dir:
            output_dir = ensure_dir(output_dir)
            results_path = Path(output_dir) / f'{model_name}_results.json'
            save_json(results, results_path)
            print(f"Saved results to {results_path}")

    # Compare models
    if len(all_results) > 1:
        metrics_dict = {name: res['metrics'] for name, res in all_results.items()}
        comparison = compare_models(metrics_dict)
        print("\nModel Comparison:")
        for metric, ranking in comparison['ranking'].items():
            print(f"  {metric}: {' > '.join(ranking)}")

        if output_dir:
            save_json(comparison, Path(output_dir) / 'model_comparison.json')

    return all_results


def save_evaluation_results(
    results: Dict[str, Any],
    output_dir: str,
    model_name: str = "model"
) -> None:
    """
    Save evaluation results to files.

    Args:
        results: Evaluation results
        output_dir: Output directory
        model_name: Model name for file naming
    """
    output_dir = ensure_dir(output_dir)

    # Save full results as JSON
    results_path = Path(output_dir) / f'{model_name}_evaluation_results.json'
    save_json(results, results_path)

    # Save metrics as CSV
    import csv
    metrics_path = Path(output_dir) / f'{model_name}_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in results['metrics'].items():
            writer.writerow([key, value])

    print(f"Saved results to {output_dir}")
