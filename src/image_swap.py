"""
Image-swap experiment for testing model robustness to image-text mismatches.
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from .dataset import MultimodalFakeNewsDataset, Sample
from .preprocessing import MultimodalPreprocessor
from .utils import set_seed, ensure_dir


@dataclass
class SwapConfig:
    """Configuration for image swap experiment."""
    num_swaps: int = 1000
    seed: int = 42
    replacement_strategy: str = "random"  # random, semantic, or fixed
    swap_within_class: bool = False  # Swap within same label class
    min_similarity_threshold: Optional[float] = None  # For semantic replacement


class ImageSwapExperiment:
    """
    Image-swap experiment to test model robustness.

    For each test sample:
    1. Keep original text and label
    2. Replace image with a different image
    3. Evaluate both original and swapped versions
    """

    def __init__(
        self,
        dataset: MultimodalFakeNewsDataset,
        config: Optional[SwapConfig] = None,
        image_pool_dir: Optional[str] = None
    ):
        """
        Initialize experiment.

        Args:
            dataset: Original dataset
            config: Swap configuration
            image_pool_dir: Directory containing replacement images
        """
        self.dataset = dataset
        self.config = config or SwapConfig()
        self.image_pool_dir = image_pool_dir

        # Set seed
        set_seed(self.config.seed)

        # Build image pool
        self.image_pool = self._build_image_pool()

    def _build_image_pool(self) -> List[str]:
        """
        Build pool of available images for replacement.

        Returns:
            List of image paths
        """
        pool = []

        # From the dataset itself
        for sample in self.dataset.samples:
            if sample.image_path and os.path.exists(sample.image_path):
                pool.append(sample.image_path)

        # From image pool directory if specified
        if self.image_pool_dir and os.path.exists(self.image_pool_dir):
            pool.extend([
                str(p) for p in Path(self.image_pool_dir).glob("**/*.jpg")
            ])
            pool.extend([
                str(p) for p in Path(self.image_pool_dir).glob("**/*.png")
            ])

        # Remove duplicates
        pool = list(set(pool))

        print(f"Built image pool with {len(pool)} images")
        return pool

    def create_swapped_dataset(
        self,
        num_samples: Optional[int] = None
    ) -> MultimodalFakeNewsDataset:
        """
        Create a dataset with swapped images.

        Args:
            num_samples: Number of samples to swap (default: all)

        Returns:
            Dataset with swapped images
        """
        num_samples = num_samples or self.config.num_swaps
        num_samples = min(num_samples, len(self.dataset.samples))

        # Shuffle and select samples to swap
        indices = list(range(len(self.dataset.samples)))
        random.shuffle(indices)
        swap_indices = indices[:num_samples]

        swapped_samples = []

        for idx, sample in enumerate(tqdm(self.dataset.samples, desc="Creating swapped dataset")):
            new_sample = Sample(
                sample_id=f"{sample.sample_id}_swapped",
                text=sample.text,
                image_path=sample.image_path,  # Will be overridden
                label=sample.label,  # Keep original label!
                metadata={
                    **sample.metadata,
                    'is_swapped': idx in swap_indices,
                    'original_image': sample.image_path
                }
            )

            if idx in swap_indices:
                # Select replacement image
                replacement = self._select_replacement_image(sample)
                new_sample.image_path = replacement
                new_sample.metadata['replacement_image'] = replacement
            else:
                new_sample.image_path = sample.image_path
                new_sample.metadata['is_swapped'] = False

            swapped_samples.append(new_sample)

        print(f"Created swapped dataset with {len(swapped_samples)} samples")
        print(f"  - {sum(1 for s in swapped_samples if s.metadata.get('is_swapped'))} swapped")
        print(f"  - {sum(1 for s in swapped_samples if not s.metadata.get('is_swapped'))} unchanged")

        return MultimodalFakeNewsDataset(samples=swapped_samples)

    def _select_replacement_image(self, sample: Sample) -> str:
        """
        Select a replacement image for swapping.

        Args:
            sample: Original sample

        Returns:
            Path to replacement image
        """
        if not self.image_pool:
            # No pool available, return original
            return sample.image_path

        # Filter pool if swapping within class
        candidates = self.image_pool.copy()

        if self.config.swap_within_class:
            # Only use images from same-label samples
            same_label_images = []
            for s in self.dataset.samples:
                if s.label == sample.label and s.image_path in candidates:
                    same_label_images.append(s.image_path)
            if same_label_images:
                candidates = same_label_images

        # Remove original image from candidates
        if sample.image_path in candidates and len(candidates) > 1:
            candidates.remove(sample.image_path)

        if not candidates:
            # Fallback to original if no candidates
            return sample.image_path

        # Select random replacement
        return random.choice(candidates)

    def run_experiment(
        self,
        model,
        preprocessor: MultimodalPreprocessor,
        device: torch.device,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Run the complete image-swap experiment.

        Args:
            model: Model to evaluate
            preprocessor: Multimodal preprocessor
            device: Device to use
            test_loader: Test data loader

        Returns:
            Dictionary with experiment results
        """
        model.eval()

        results = {
            'original': [],
            'swapped': [],
            'swapped_indices': [],
            'unchanged_indices': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running swap experiment")):
                texts = batch['text']
                images = batch.get('image', [None] * len(texts))
                labels = torch.tensor(batch['label'], dtype=torch.float32).to(device)
                is_swapped = batch.get('is_swapped', [False] * len(texts))

                # Preprocess
                inputs = preprocessor.preprocess_batch(texts, images)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get predictions
                logits, intermediate = model(**inputs)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

                # Store results
                for i in range(len(labels)):
                    sample_result = {
                        'label': labels[i].item(),
                        'prob_fake': probs[i],
                        'pred': 1 if probs[i] >= 0.5 else 0
                    }

                    if is_swapped[i]:
                        results['swapped'].append(sample_result)
                        results['swapped_indices'].append(batch_idx * len(labels) + i)
                    else:
                        results['original'].append(sample_result)
                        results['unchanged_indices'].append(batch_idx * len(labels) + i)

        # Compute summary statistics
        summary = self._compute_summary(results)

        return {
            'results': results,
            'summary': summary,
            'config': self.config.__dict__
        }

    def _compute_summary(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        original_results = results['original']
        swapped_results = results['swapped']

        def compute_metrics(r_list):
            if not r_list:
                return {}
            probs = np.array([r['prob_fake'] for r in r_list])
            preds = (probs >= 0.5).astype(int)
            labels = np.array([r['label'] for r in r_list])

            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            return {
                'n_samples': len(r_list),
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1_score(labels, preds, zero_division=0),
                'mean_prob_fake': np.mean(probs),
                'std_prob_fake': np.std(probs)
            }

        summary = {
            'original': compute_metrics(original_results),
            'swapped': compute_metrics(swapped_results),
            'n_swapped': len(swapped_results),
            'n_unchanged': len(original_results)
        }

        # Compute differences
        if summary['original'] and summary['swapped']:
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'mean_prob_fake']:
                if metric in summary['original'] and metric in summary['swapped']:
                    summary[f'{metric}_difference'] = (
                        summary['swapped'][metric] - summary['original'][metric]
                    )

        return summary


def controlled_swap_evaluation(
    model,
    preprocessor: MultimodalPreprocessor,
    original_loader: DataLoader,
    swapped_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Compare model performance on original vs swapped pairs.

    Args:
        model: Model to evaluate
        preprocessor: Multimodal preprocessor
        original_loader: Original test data
        swapped_loader: Swapped test data
        device: Device

    Returns:
        Comparison results
    """
    from .evaluate import Evaluator

    evaluator = Evaluator(model, preprocessor, device)

    print("Evaluating on original pairs...")
    original_results = evaluator.evaluate(original_loader)

    print("Evaluating on swapped pairs...")
    swapped_results = evaluator.evaluate(swapped_loader)

    # Compute comparison
    comparison = {
        'original_metrics': original_results['metrics'],
        'swapped_metrics': swapped_results['metrics'],
        'original_n': original_results['metrics']['n_samples'],
        'swapped_n': swapped_results['metrics']['n_samples'],
        'differences': {}
    }

    for metric in ['accuracy', 'macro_f1', 'precision', 'recall', 'roc_auc']:
        orig = original_results['metrics'].get(metric, 0)
        swap = swapped_results['metrics'].get(metric, 0)
        comparison['differences'][metric] = swap - orig

    # Prediction probability changes
    orig_probs = np.array(original_results['probabilities'])
    swap_probs = np.array(swapped_results['probabilities'])

    comparison['probability_changes'] = {
        'mean_change': float(np.mean(swap_probs - orig_probs)),
        'std_change': float(np.std(swap_probs - orig_probs)),
        'max_increase': float(np.max(swap_probs - orig_probs)),
        'max_decrease': float(np.min(swap_probs - orig_probs))
    }

    # Count prediction flips
    orig_preds = (orig_probs >= 0.5).astype(int)
    swap_preds = (swap_probs >= 0.5).astype(int)
    flips = orig_preds != swap_preds

    comparison['prediction_flips'] = {
        'total_flips': int(np.sum(flips)),
        'flip_rate': float(np.mean(flips)),
        'flips_to_fake': int(np.sum((~flips.astype(bool)) & swap_preds.astype(bool))),
        'flips_to_real': int(np.sum(flips.astype(bool) & (~swap_preds.astype(bool))))
    }

    return comparison
