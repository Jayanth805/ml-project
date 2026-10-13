"""
Script to train multimodal fake news detection models.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils import load_config, set_seed, get_device, ensure_dir
from src.dataset import MultimodalFakeNewsDataset, Sample
from src.preprocessing import MultimodalPreprocessor
from src.classifier import MultimodalClassifier, ClassifierConfig
from src.train import Trainer, TrainingConfig
from torch.utils.data import DataLoader, default_collate
import pandas as pd


def multimodal_collate(batch):
    """Collate samples while allowing unavailable images to remain None."""
    collated = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        collated[key] = values if key == "image" else default_collate(values)
    return collated


def train_model(
    model_type: str,
    config_path: str = "configs/experiment.yaml",
    data_dir: str = "data/processed",
    images_dir: str = "data/fakeddit/images",
    output_dir: str = "artifacts/checkpoints",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    init_checkpoint: str = None
):
    """
    Train a multimodal fake news detection model.

    Args:
        model_type: Type of model to train
        config_path: Path to config file
        data_dir: Path to processed data directory
        images_dir: Path to images directory
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seed: Random seed
    """
    print("=" * 60)
    print(f"Training Model: {model_type}")
    print("=" * 60)

    # Set seed
    set_seed(seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # Create output directory
    output_path = Path(output_dir) / model_type
    ensure_dir(output_path)

    # Load datasets
    print(f"\nLoading datasets from {data_dir}...")
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    val_df = pd.read_csv(Path(data_dir) / "val.csv")

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

    train_samples = create_samples(train_df, images_dir)
    val_samples = create_samples(val_df, images_dir)

    train_dataset = MultimodalFakeNewsDataset(train_samples)
    val_dataset = MultimodalFakeNewsDataset(val_samples)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=multimodal_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=multimodal_collate
    )

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = MultimodalPreprocessor(
        text_model_name=config.get('model', {}).get('text_encoder', {}).get('name', 'google-bert/bert-base-uncased'),
        image_model_name=config.get('model', {}).get('image_encoder', {}).get('name', 'google/vit-base-patch16-224'),
        clip_model_name=config.get('model', {}).get('clip', {}).get('name', 'openai/clip-vit-base-patch32'),
        max_text_length=config.get('dataset', {}).get('max_text_length', 128),
        device=device
    )

    # Create model
    print(f"\nCreating {model_type} model...")
    model_config = ClassifierConfig(
        model_type=model_type,
        freeze_encoders=True,
        projection_dim=config.get('model', {}).get('fusion', {}).get('projection_dim', 256),
        fusion_hidden_dim=config.get('model', {}).get('fusion', {}).get('hidden_dim', 256),
        dropout=config.get('model', {}).get('fusion', {}).get('dropout', 0.3)
    )

    model = MultimodalClassifier(model_config)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Initialized weights from {init_checkpoint}")
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    train_config = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=config.get('training', {}).get('weight_decay', 1e-4),
        gradient_clip_norm=config.get('training', {}).get('gradient_clip_norm', 1.0),
        early_stopping_patience=config.get('training', {}).get('early_stopping_patience', 3),
        device=str(device),
        checkpoint_dir=str(output_path),
        log_interval=10,
        save_best_only=True,
        seed=seed
    )

    trainer = Trainer(
        model=model,
        preprocessor=preprocessor,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Train
    print("\nStarting training...")
    history = trainer.train()

    # Save final checkpoint
    trainer.save_checkpoint(history.best_epoch, is_best=False, filename=f"{model_type}_final.pt")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation F1: {history.best_val_f1:.4f}")
    print(f"Best epoch: {history.best_epoch + 1}")
    print(f"Checkpoints saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train multimodal fake news detection model")
    parser.add_argument("--model", required=True,
                        choices=['text_only', 'image_only', 'standard', 'consistency_aware', 'ablation'],
                        help="Type of model to train")
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="Path to config file")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Path to processed data directory")
    parser.add_argument("--images-dir", default="data/fakeddit/images",
                        help="Path to images directory")
    parser.add_argument("--output-dir", default="artifacts/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Optional checkpoint to continue training from")

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        config_path=args.config,
        data_dir=args.data_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint
    )


if __name__ == "__main__":
    main()
