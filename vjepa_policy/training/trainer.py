"""
Training Loop

Handles training of the V-JEPA 2 policy model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Trainer for V-JEPA 2 policy.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        device: str = "cuda",
        log_every: int = 10,
        save_dir: str = "checkpoints",
        use_wandb: bool = True,
        use_precomputed: bool = False,
        seed: int = 42,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_every = log_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_precomputed = use_precomputed
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.lr = lr

        # Set seed
        set_seed(seed)

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler will be created in train() with correct T_max
        self.scheduler = None
        self.global_step = 0
        self.best_val_loss = float('inf')

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Uses L1 loss (smoother than L2 for actions).
        """
        loss = F.l1_loss(pred_actions, target_actions)

        # Per-dimension loss for debugging
        loss_per_dim = F.l1_loss(pred_actions, target_actions, reduction='none').mean(dim=(0, 1))

        return {
            'loss': loss,
            'loss_per_dim': loss_per_dim,
        }

    def warmup_lr(self, epoch: int, warmup_epochs: int):
        """Linear warmup for learning rate"""
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr * warmup_factor

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move to device
            if self.use_precomputed:
                current_emb = batch['current_emb'].to(self.device)
                goal_emb = batch['goal_emb'].to(self.device)
                proprio = batch['proprio'].to(self.device)
                target_actions = batch['actions'].to(self.device)

                # Forward with precomputed embeddings
                pred_actions = self.model.forward_with_precomputed(
                    current_emb, goal_emb, proprio
                )
            else:
                video = batch['video'].to(self.device)
                goal = batch['goal'].to(self.device)
                proprio = batch['proprio'].to(self.device)
                target_actions = batch['actions'].to(self.device)

                # Forward pass
                pred_actions = self.model(video, goal, proprio)

            # Compute loss
            losses = self.compute_loss(pred_actions, target_actions)
            loss = losses['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_params(),
                    self.grad_clip
                )

            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            n_batches += 1

            if self.global_step % self.log_every == 0:
                pbar.set_postfix({'loss': loss.item()})

                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/step': self.global_step,
                    })

            self.global_step += 1

        avg_loss = total_loss / n_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            if self.use_precomputed:
                current_emb = batch['current_emb'].to(self.device)
                goal_emb = batch['goal_emb'].to(self.device)
                proprio = batch['proprio'].to(self.device)
                target_actions = batch['actions'].to(self.device)

                pred_actions = self.model.forward_with_precomputed(
                    current_emb, goal_emb, proprio
                )
            else:
                video = batch['video'].to(self.device)
                goal = batch['goal'].to(self.device)
                proprio = batch['proprio'].to(self.device)
                target_actions = batch['actions'].to(self.device)

                pred_actions = self.model(video, goal, proprio)

            losses = self.compute_loss(pred_actions, target_actions)
            total_loss += losses['loss'].item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        if self.use_wandb:
            wandb.log({'val/loss': avg_loss})

        return {'loss': avg_loss}

    def train(self, n_epochs: int):
        """Full training loop"""

        # Create scheduler with correct T_max
        if self.warmup_epochs < n_epochs:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=n_epochs - self.warmup_epochs,
                eta_min=self.lr * 0.01,  # Minimum LR
            )

        print(f"\nStarting training for {n_epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Warmup epochs: {self.warmup_epochs}")
        print()

        for epoch in range(n_epochs):
            # Warmup learning rate
            if epoch < self.warmup_epochs:
                self.warmup_lr(epoch, self.warmup_epochs)

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch}: train_loss = {train_metrics['loss']:.4f}")

            # Update scheduler (after warmup)
            if epoch >= self.warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                print(f"Epoch {epoch}: val_loss = {val_metrics['loss']:.4f}")

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint("best_model.pt")
                    print(f"  New best model saved!")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

        # Save final model
        self.save_checkpoint("final_model.pt")
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.save_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint: {path}")
