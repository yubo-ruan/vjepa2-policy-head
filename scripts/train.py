#!/usr/bin/env python3
"""
Unified Training Script for V-JEPA 2 Policy.

Single training script with CLI flags for all options.
Works with precomputed spatial embeddings only.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --save_dir /workspace/checkpoints
    python scripts/train.py --config configs/config.yaml --epochs 50 --lr 0.0001
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.policy import VJEPA2Policy
from vjepa_policy.data.dataset import create_dataloaders
from vjepa_policy.training.loss import create_loss
from vjepa_policy.training.trainer import Trainer

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train V-JEPA 2 Policy')

    # Config
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    # Overrides
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

    # Data
    parser.add_argument('--suite', type=str, default=None,
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10'],
                        help='Override LIBERO suite')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                        help='Override embeddings directory')

    # Gripper augmentation
    parser.add_argument('--gripper_oversample', type=int, default=None,
                        help='Gripper transition oversampling factor (1=disabled)')
    parser.add_argument('--gripper_jitter', type=int, default=None,
                        help='Gripper jitter range in timesteps (0=disabled)')

    # Logging
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
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

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print config
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

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config.get('logging', {}).get('wandb_project', 'vjepa2-policy'),
            config=config,
            name=args.run_name or f"{config['data'].get('suite', 'libero')}_training",
        )

    # Create dataloaders
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

    # Create model
    print("Creating model...")
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
        device=device,
    )

    # Print parameter counts
    counts = model.count_parameters()
    print("Parameters:")
    for name, count in counts.items():
        print(f"  {name}: {count / 1e6:.2f}M")
    print()

    # Create loss
    loss_fn = create_loss(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device,
        save_dir=save_dir,
        use_wandb=use_wandb,
    )

    # Train
    trainer.train(config['training']['epochs'])

    if use_wandb:
        wandb.finish()

    print()
    print(f"Training complete!")
    print(f"Best model: {Path(save_dir) / 'best_model.pt'}")


if __name__ == '__main__':
    main()
