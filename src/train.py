"""
Training pipeline for multimodal fake news detection models.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .classifier import MultimodalClassifier, ClassifierConfig
from .preprocessing import MultimodalPreprocessor
from .utils import AverageMeter, set_seed, ensure_dir, timestamp_now
from .metrics import compute_metrics, print_metrics


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = False
    checkpoint_dir: str = "artifacts/checkpoints"
    log_interval: int = 10
    save_best_only: bool = True


@dataclass
class TrainingHistory:
    """Stores training history."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_f1: float = 0.0


class Trainer:
    """
    Trainer for multimodal fake news classification models.
    """

    def __init__(
        self,
        model: MultimodalClassifier,
        preprocessor: MultimodalPreprocessor,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            preprocessor: Multimodal preprocessor
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optional custom optimizer
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        if optimizer is None:
            # Only train fusion parameters (encoders are frozen)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize F1
                factor=0.5,
                patience=2
            )
        else:
            self.scheduler = scheduler

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision
        self.use_amp = config.use_amp
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # History
        self.history = TrainingHistory()

        # Checkpoint directory
        self.checkpoint_dir = ensure_dir(config.checkpoint_dir)

        # Set seed
        set_seed(config.seed)

    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        loss_meter = AverageMeter()
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Prepare inputs
            texts = batch['text']
            images = [b if b is not None else None for b in batch.get('image', [None]*len(batch['text']))]
            labels = batch['label'].to(device=self.device, dtype=torch.float32)

            # Preprocess
            inputs = self.preprocessor.preprocess_batch(texts, images)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(**inputs)
                    loss = self.criterion(logits.squeeze(-1), labels)
            else:
                logits, _ = self.model(**inputs)
                loss = self.criterion(logits.squeeze(-1), labels)

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.gradient_clip_norm
                )
                self.optimizer.step()

            # Track metrics
            loss_meter.update(loss.item(), len(labels))
            probs = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

        # Compute metrics
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        return loss_meter.avg, metrics

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        loss_meter = AverageMeter()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Prepare inputs
                texts = batch['text']
                images = [b if b is not None else None for b in batch.get('image', [None]*len(batch['text']))]
                labels = batch['label'].to(device=self.device, dtype=torch.float32)

                # Preprocess
                inputs = self.preprocessor.preprocess_batch(texts, images)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                logits, _ = self.model(**inputs)
                loss = self.criterion(logits.squeeze(-1), labels)

                # Track metrics
                loss_meter.update(loss.item(), len(labels))
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        # Compute metrics
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        return loss_meter.avg, metrics

    def train(self) -> TrainingHistory:
        """
        Train the model for multiple epochs.

        Returns:
            TrainingHistory object
        """
        print(f"\nStarting training on {self.device}")
        print(f"Model: {self.model.model_type}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")

        patience_counter = 0
        best_f1 = 0.0

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.history.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.history.val_losses.append(val_loss)
            self.history.val_metrics.append(val_metrics)

            # Update scheduler
            self.scheduler.step(val_metrics.get('macro_f1', 0))

            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: {val_metrics}")

            # Save checkpoint
            is_best = val_metrics.get('macro_f1', 0) > best_f1
            if is_best:
                best_f1 = val_metrics.get('macro_f1', 0)
                self.history.best_epoch = epoch
                self.history.best_val_f1 = best_f1

            if self.config.save_best_only:
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  Saved best model (F1: {best_f1:.4f})")
            else:
                self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if epoch > 0:
                if val_loss >= self.history.best_val_loss:
                    patience_counter += 1
                else:
                    patience_counter = 0

                self.history.best_val_loss = min(val_loss, self.history.best_val_loss)

                if patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        print(f"\nTraining complete. Best F1: {self.history.best_val_f1:.4f}")
        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
            filename: Optional custom filename

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"{self.model.model_type}_epoch_{epoch+1}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history.__dict__,
            'config': {
                'model_config': self.model.config.__dict__,
                'training_config': self.config.__dict__
            },
            'timestamp': timestamp_now()
        }

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.model.model_type}_best.pt"
            torch.save(checkpoint, best_path)

        return str(checkpoint_path)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        preprocessor: MultimodalPreprocessor,
        device: Optional[torch.device] = None
    ) -> Tuple[MultimodalClassifier, TrainingHistory, Dict[str, Any]]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            preprocessor: Preprocessor for inference
            device: Device to load to

        Returns:
            Tuple of (model, history, config)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Reconstruct model
        model_config = ClassifierConfig(**checkpoint['config']['model_config'])
        model = MultimodalClassifier(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Reconstruct history
        history = TrainingHistory(**checkpoint['history'])

        return model, history, checkpoint['config']


def extract_features_batch(
    model: MultimodalClassifier,
    preprocessor: MultimodalPreprocessor,
    data_loader: DataLoader,
    device: torch.device,
    cache_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features for all samples in a dataloader.
    Used for caching features to speed up training.

    Args:
        model: Trained model with frozen encoders
        preprocessor: Multimodal preprocessor
        data_loader: Data loader
        device: Device to use
        cache_path: Optional path to save cached features

    Returns:
        Tuple of (text_features, image_features, labels)
    """
    model.eval()

    all_text_features = []
    all_image_features = []
    all_clip_similarities = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            texts = batch['text']
            images = [b if b is not None else None for b in batch.get('image', [None]*len(batch['text']))]
            labels = torch.tensor(batch['label'], dtype=torch.float32)

            # Preprocess
            inputs = preprocessor.preprocess_batch(texts, images)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Extract features
            features = model.encoders(**inputs)
            all_text_features.append(features['text_features'].cpu())
            all_image_features.append(features['image_features'].cpu())
            all_clip_similarities.append(features['clip_similarity'].cpu())
            all_labels.append(labels)

    text_features = torch.cat(all_text_features, dim=0)
    image_features = torch.cat(all_image_features, dim=0)
    clip_similarities = torch.cat(all_clip_similarities, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Cache features
    if cache_path:
        torch.save({
            'text_features': text_features,
            'image_features': image_features,
            'clip_similarities': clip_similarities,
            'labels': labels
        }, cache_path)
        print(f"Cached features to {cache_path}")

    return text_features, image_features, clip_similarities, labels
