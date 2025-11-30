"""
Simplified Training Loop for V-JEPA 2 Policy.

Clean training loop that works with precomputed spatial embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import time

from .loss import ActionLoss, create_loss


# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Simplified trainer for V-JEPA 2 policy.

    Works with precomputed spatial embeddings only.

    Args:
        model: Policy model (VJEPA2Policy or similar)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        loss_fn: Loss function (optional, created from config if not provided)
        config: Configuration dict
        device: Device to train on
        save_dir: Directory to save checkpoints
        use_wandb: Enable W&B logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[ActionLoss] = None,
        config: Optional[dict] = None,
        device: str = 'cuda',
        save_dir: str = 'checkpoints',
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.config = config or {}

        # Get training config
        train_cfg = self.config.get('training', {})
        self.lr = train_cfg.get('lr', 1e-4)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 5)
        self.grad_clip = train_cfg.get('grad_clip', 1.0)
        self.save_every = train_cfg.get('save_every', 10)
        self.log_every = train_cfg.get('log_every', 10)

        # Set seed
        seed = train_cfg.get('seed', 42)
        set_seed(seed)

        # Loss function
        if loss_fn is None:
            loss_fn = create_loss(self.config)
        self.loss_fn = loss_fn

        # Optimizer - get trainable params from model
        if hasattr(model, 'get_trainable_params'):
            params = model.get_trainable_params()
        else:
            params = model.parameters()

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=train_cfg.get('weight_decay', 1e-5),
        )

        # Scheduler (created in train() with correct T_max)
        self.scheduler = None

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Enable training mode on dataset if available
        if hasattr(self.train_loader.dataset, 'train'):
            self.train_loader.dataset.train()

        total_loss = 0
        total_pos_loss = 0
        total_grip_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Get batch data - spatial tokens
            video_tokens = batch['video_tokens'].to(self.device)
            goal_tokens = batch['goal_tokens'].to(self.device)
            proprio = batch['proprio'].to(self.device)
            actions = batch['actions'].to(self.device)
            weights = batch.get('weights')
            if weights is not None:
                weights = weights.to(self.device)

            # Forward pass
            if hasattr(self.model, 'forward_with_precomputed'):
                pred = self.model.forward_with_precomputed(video_tokens, goal_tokens, proprio)
            else:
                pred = self.model(video_tokens, goal_tokens, proprio)

            # Compute loss
            losses = self.loss_fn(pred, actions, weights)
            loss = losses['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                if hasattr(self.model, 'get_trainable_params'):
                    params = self.model.get_trainable_params()
                else:
                    params = self.model.parameters()
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            total_pos_loss += losses.get('pos_loss', torch.tensor(0)).item()
            total_grip_loss += losses.get('gripper_loss', torch.tensor(0)).item()
            n_batches += 1

            pbar.set_postfix({'loss': loss.item()})

            if self.use_wandb and self.global_step % self.log_every == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/pos_loss': losses.get('pos_loss', torch.tensor(0)).item(),
                    'train/gripper_loss': losses.get('gripper_loss', torch.tensor(0)).item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step,
                })

            self.global_step += 1

        return {
            'loss': total_loss / n_batches,
            'pos_loss': total_pos_loss / n_batches,
            'gripper_loss': total_grip_loss / n_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        # Enable eval mode on dataset if available
        if hasattr(self.val_loader.dataset, 'eval'):
            self.val_loader.dataset.eval()

        total_loss = 0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            video_tokens = batch['video_tokens'].to(self.device)
            goal_tokens = batch['goal_tokens'].to(self.device)
            proprio = batch['proprio'].to(self.device)
            actions = batch['actions'].to(self.device)

            # Forward pass
            if hasattr(self.model, 'forward_with_precomputed'):
                pred = self.model.forward_with_precomputed(video_tokens, goal_tokens, proprio)
            else:
                pred = self.model(video_tokens, goal_tokens, proprio)

            # Compute loss (no weights for validation)
            losses = self.loss_fn(pred, actions)
            total_loss += losses['loss'].item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if self.use_wandb:
            wandb.log({'val/loss': avg_loss})

        return {'loss': avg_loss}

    def warmup_lr(self, epoch: int):
        """Linear warmup for learning rate."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr * warmup_factor

    def save_checkpoint(self, filename: str, epoch: int = 0, val_loss: float = 0):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.save_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint: {path}")
        return checkpoint.get('epoch', 0)

    def train(self, epochs: int):
        """Full training loop."""

        # Create scheduler
        if self.warmup_epochs < epochs:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - self.warmup_epochs,
                eta_min=self.lr * 0.01,
            )

        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Save directory: {self.save_dir}")
        print("=" * 60)
        print()

        for epoch in range(epochs):
            start_time = time.time()

            # Warmup learning rate
            if epoch < self.warmup_epochs:
                self.warmup_lr(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)

            # Update scheduler (after warmup)
            if epoch >= self.warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            # Validate
            val_metrics = self.validate()

            elapsed = time.time() - start_time

            # Check for best model
            is_best = False
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                is_best = True
                self.save_checkpoint('best_model.pt', epoch, val_metrics['loss'])

            # Save periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}.pt', epoch, val_metrics.get('loss', 0))

            # Save latest
            self.save_checkpoint('latest.pt', epoch, val_metrics.get('loss', 0))

            # Print progress
            val_str = f" | Val: {val_metrics['loss']:.4f}" if val_metrics else ""
            best_str = " [BEST]" if is_best else ""
            print(f"Epoch {epoch:3d} | Train: {train_metrics['loss']:.4f}{val_str} | Time: {elapsed:.1f}s{best_str}")

        print()
        print("=" * 60)
        print("Training Complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best model: {self.save_dir / 'best_model.pt'}")
        print("=" * 60)
