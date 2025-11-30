#!/usr/bin/env python3
"""
Unified Training Script for V-JEPA 2 Policy.

This script trains a goal-conditioned manipulation policy using precomputed
V-JEPA 2 spatial embeddings. It's the main entry point for training.

Architecture:
    - Uses precomputed spatial tokens (64 per modality, 8x8 grid)
    - Transformer decoder policy head with cross-attention
    - Action chunking (50 timesteps per prediction)
    - Weighted loss with gripper and temporal emphasis

Training Pipeline:
    1. Load precomputed embeddings from disk
    2. Create PolicyDataset with augmentations (noise, gripper oversampling)
    3. Initialize VJEPA2Policy model (V-JEPA 2 encoder is lazy-loaded, not used)
    4. Train with AdamW optimizer and cosine LR schedule
    5. Save best model based on validation loss

Prerequisites:
    Run precompute.py first to generate spatial embeddings:
        python scripts/precompute.py --suite libero_spatial --output_dir /workspace/data/embeddings

Usage:
    # Basic training with default config
    python scripts/train.py --config configs/config.yaml

    # Override specific settings
    python scripts/train.py --config configs/config.yaml --epochs 50 --lr 0.0001

    # Enable gripper-focused augmentation
    python scripts/train.py --config configs/config.yaml --gripper_oversample 5 --gripper_jitter 5

    # Custom save directory
    python scripts/train.py --config configs/config.yaml --save_dir /workspace/checkpoints/exp1

CLI Arguments:
    --config: Path to YAML config file (required)
    --save_dir: Override checkpoint save directory
    --epochs: Override number of training epochs
    --lr: Override learning rate
    --batch_size: Override batch size
    --seed: Override random seed
    --suite: Override LIBERO suite (libero_spatial, libero_object, etc.)
    --embeddings_dir: Override embeddings directory
    --gripper_oversample: Oversample gripper transitions (1=disabled)
    --gripper_jitter: Temporal jitter range for transitions (0=disabled)
    --no_wandb: Disable Weights & Biases logging
    --run_name: Custom W&B run name
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.policy import VJEPA2Policy
from vjepa_policy.data.dataset import create_dataloaders
from vjepa_policy.training.loss import create_loss
from vjepa_policy.training.trainer import Trainer

# Optional wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(description='Train V-JEPA 2 Policy')

    # Config file (primary source of settings)
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    # Override options (take precedence over config file)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override save directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed')

    # Data options
    parser.add_argument('--suite', type=str, default=None,
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10'],
                        help='Override LIBERO suite')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                        help='Override embeddings directory')

    # Gripper augmentation (key for learning precise gripper timing)
    parser.add_argument('--gripper_oversample', type=int, default=None,
                        help='Gripper transition oversampling factor (1=disabled)')
    parser.add_argument('--gripper_jitter', type=int, default=None,
                        help='Gripper jitter range in timesteps (0=disabled)')

    # Logging options
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (or "latest" to use latest.pt)')

    args = parser.parse_args()

    # ========== Load and Merge Config ==========
    config = load_config(args.config)

    # Apply CLI overrides (CLI args take precedence)
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.suite is not None:
        config['data']['suite'] = args.suite
    if args.embeddings_dir is not None:
        config['data']['embeddings_dir'] = args.embeddings_dir
    if args.gripper_oversample is not None:
        config['data']['gripper_oversample'] = args.gripper_oversample
    if args.gripper_jitter is not None:
        config['data']['gripper_jitter'] = args.gripper_jitter

    save_dir = args.save_dir or config.get('logging', {}).get('checkpoint_dir', 'checkpoints')

    # ========== Setup ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print configuration summary
    print("=" * 60)
    print("V-JEPA 2 Policy Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")
    print()
    print("Model:")
    for k, v in config['model'].items():
        print(f"  {k}: {v}")
    print()
    print("Data:")
    for k, v in config['data'].items():
        print(f"  {k}: {v}")
    print()
    print("Training:")
    for k, v in config['training'].items():
        print(f"  {k}: {v}")
    print()
    print("Loss:")
    for k, v in config.get('loss', {}).items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print()

    # ========== W&B Settings ==========
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    wandb_project = config.get('logging', {}).get('wandb_project', 'vjepa2-policy') if use_wandb else None
    wandb_run_name = args.run_name or f"{config['data'].get('suite', 'libero')}_training"

    # ========== Create Dataloaders ==========
    print("Loading data...")
    try:
        train_loader, val_loader = create_dataloaders(config)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print()
        print("Please precompute embeddings first:")
        print(f"  python scripts/precompute.py --suite {config['data'].get('suite', 'libero_spatial')}")
        return
    print()

    # ========== Create Model ==========
    # Note: V-JEPA 2 encoder is lazy-loaded and NOT used during training
    # (we use precomputed embeddings). It's only loaded during evaluation.
    use_goal_conditioned = config['model'].get('use_goal_conditioned', False)
    print("Creating model...")
    if use_goal_conditioned:
        print("  Using GoalConditionedPolicyHead (goal-dependent action queries)")
    model = VJEPA2Policy(
        vjepa2_model_path=config.get('encoder', {}).get('model_path', '/workspace/models/vjepa2-ac-vitg.pt'),
        vjepa2_model_name=config.get('encoder', {}).get('model_name', 'vjepa2_vitg'),
        vjepa2_freeze=config.get('encoder', {}).get('freeze', True),
        vjepa2_num_frames=config.get('encoder', {}).get('num_frames', 16),
        proprio_dim=config['model'].get('proprio_dim', 15),
        proprio_history=config['model'].get('proprio_history', 5),
        embed_dim=config['model'].get('embed_dim', 1408),
        hidden_dim=config['model'].get('hidden_dim', 512),
        num_heads=config['model'].get('num_heads', 8),
        num_layers=config['model'].get('num_layers', 4),
        num_spatial_tokens=config['model'].get('num_spatial_tokens', 64),
        action_dim=config['model'].get('action_dim', 7),
        chunk_size=config['model'].get('chunk_size', 50),
        dropout=config['model'].get('dropout', 0.1),
        use_goal_conditioned=use_goal_conditioned,
        device=device,
    )

    # Print parameter counts
    counts = model.count_parameters()
    print("Parameters:")
    for name, count in counts.items():
        print(f"  {name}: {count / 1e6:.2f}M")
    print()

    # ========== Create Loss and Trainer ==========
    loss_fn = create_loss(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
        save_dir=save_dir,
        use_wandb=False,  # Let train() initialize W&B
    )

    # ========== Resume from checkpoint if requested ==========
    start_epoch = 0
    if args.resume:
        if args.resume == 'latest':
            resume_path = Path(save_dir) / 'latest.pt'
        else:
            resume_path = Path(args.resume)

        if resume_path.exists():
            start_epoch = trainer.load_checkpoint(str(resume_path))
            # The checkpoint stores the epoch that completed, so we start from next
            start_epoch = start_epoch + 1
            print(f"Resuming from epoch {start_epoch}")

            # Recreate scheduler with remaining epochs
            remaining_epochs = config['training']['epochs'] - start_epoch
            if remaining_epochs > 0 and trainer.warmup_epochs < config['training']['epochs']:
                trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    trainer.optimizer,
                    T_max=remaining_epochs,
                    eta_min=trainer.lr * 0.01,
                )
        else:
            print(f"Warning: Checkpoint not found at {resume_path}, starting from scratch")

    # ========== Train ==========
    trainer.train(
        epochs=config['training']['epochs'],
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        start_epoch=start_epoch,
    )

    print()
    print(f"Training complete!")
    print(f"Best model: {Path(save_dir) / 'best_model.pt'}")


if __name__ == '__main__':
    main()
