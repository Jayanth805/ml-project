"""
Script to evaluate trained models.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np

from src.utils import load_config, set_seed, get_device, ensure_dir, save_json
from src.dataset import MultimodalFakeNewsDataset, Sample
from src.preprocessing import MultimodalPreprocessor
from src.classifier import MultimodalClassifier, ClassifierConfig
from src.evaluate import Evaluator
from src.metrics import print_metrics
from torch.utils.data import DataLoader, default_collate


def multimodal_collate(batch):
    """Keep optional image values as a list so missing images can be handled downstream."""
    return {
        key: ([sample[key] for sample in batch] if key == "image"
              else default_collate([sample[key] for sample in batch]))
        for key in batch[0]
    }


def evaluate_model(
    checkpoint_path: str,
    config_path: str = "configs/experiment.yaml",
    data_dir: str = "data/processed",
    images_dir: str = "data/fakeddit/images",
    output_dir: str = "results",
    model_type: str = None,
    batch_size: int = 4,
    num_workers: int = 0
):
    """
    Evaluate a trained model on the test set.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        data_dir: Path to processed data directory
        images_dir: Path to images directory
        output_dir: Output directory for results
        model_type: Model type (auto-detected from checkpoint if not specified)
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Set seed
    set_seed(42)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type is None:
        model_type = checkpoint.get('config', {}).get('model_config', {}).get('model_type', 'unknown')

    print(f"Model type: {model_type}")

    # Reconstruct model
    model_config = ClassifierConfig(**checkpoint['config']['model_config'])
    model = MultimodalClassifier(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    print(f"\nLoading test data from {data_dir}...")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    def create_samples(df, images_dir):
        samples = []
        for _, row in df.iterrows():
            text = str(row.get('text', row.iloc[0]))
            image_id = str(row.get('image_id', ''))
            label = int(row.get('label', row.iloc[-1]))

            image_path = Path(images_dir) / f"{image_id}.jpg" if image_id else None

            sample = Sample(
                sample_id=str(row.name),
                text=text,
                image_path=str(image_path) if image_path and image_path.exists() else None,
                label=label,
                metadata={}
            )
            samples.append(sample)
        return samples

    test_samples = create_samples(test_df, images_dir)
    test_dataset = MultimodalFakeNewsDataset(test_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=multimodal_collate
    )

    print(f"Test samples: {len(test_dataset)}")

    # Initialize preprocessor
    preprocessor = MultimodalPreprocessor(device=device)

    # Create evaluator
    evaluator = Evaluator(model, preprocessor, device)

    # Evaluate
    print("\nEvaluating...")
    results = evaluator.evaluate(test_loader)

    # Print metrics
    print_metrics(results['metrics'], f"{model_type} Results")

    # Generate plots
    output_path = Path(output_dir)
    ensure_dir(output_path)

    print("\nGenerating plots...")
    plot_paths = evaluator.generate_plots(results, str(output_path / 'plots'), model_type)

    # Save results
    save_json(results, output_path / f'{model_type}_results.json')

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="Path to config file")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--images-dir", default="data/fakeddit/images",
                        help="Path to images directory")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--model-type",
                        choices=['text_only', 'image_only', 'standard', 'consistency_aware', 'ablation'],
                        help="Model type (auto-detected if not specified)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Evaluation batch size (lower values reduce CPU memory use)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Data loader workers (0 avoids extra worker memory on CPU)")

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=args.data_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
