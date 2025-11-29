#!/usr/bin/env python3
"""
Training script with gripper-focused data augmentation.

Addresses the gripper timing issue where the model closes the gripper
~20-30 steps too early by:

1. Oversampling gripper transitions (5×)
2. Temporal jitter (±5 steps) for transition timing variety
3. Weighted loss with class balancing
4. Extra focus on gripper dimension

Usage:
    python scripts/train_gripper_focused.py \
        --config configs/spatial.yaml \
        --suite libero_spatial \
        --oversample_factor 5 \
        --jitter_range 5 \
        --save_dir /workspace/checkpoints_gripper_v1
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2PolicySpatial
from vjepa_policy.data.libero_dataset import GripperFocusedSpatialDataset
from vjepa_policy.training.trainer import Trainer
from torch.utils.data import DataLoader

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.suite:
        config['data']['suite'] = args.suite
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr

    suite = config['data']['suite']

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check embeddings exist
    embedding_dir = Path(args.embedding_dir) / f"{suite}_spatial"
    if not embedding_dir.exists():
        print(f"Error: Spatial embeddings not found at {embedding_dir}")
        print(f"Please run: python scripts/precompute_spatial_embeddings.py --suite {suite}")
        return

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            config={
                **config,
                'gripper_augmentation': {
                    'oversample_factor': args.oversample_factor,
                    'jitter_range': args.jitter_range,
                    'transition_weight': args.transition_weight,
                    'gripper_loss_weight': args.gripper_loss_weight,
                    'balance_gripper': args.balance_gripper,
                }
            },
            name=args.run_name or f"{suite}_gripper_focused",
        )

    # Get embedding dimension from a sample
    sample_path = list((embedding_dir / "train").glob("*.pt"))[0]
    sample_data = torch.load(sample_path)
    n_tokens, embed_dim = sample_data['video_emb'].shape
    print(f"Detected spatial tokens: ({n_tokens}, {embed_dim})")

    # Create model
    print("Creating VJEPA2PolicySpatial model...")
    model = VJEPA2PolicySpatial(
        vjepa2_model_path=config['vjepa2']['model_path'],
        vjepa2_model_name=config['vjepa2']['model_name'],
        vjepa2_freeze=True,
        vjepa2_num_frames=config['vjepa2']['num_frames'],
        proprio_dim=config['proprio']['dim'],
        proprio_history=config['proprio']['history_len'],
        proprio_output_dim=config['proprio'].get('output_dim', 256),
        policy_hidden_dim=config['policy']['hidden_dim'],
        policy_n_heads=config['policy']['n_heads'],
        policy_n_layers=config['policy']['n_layers'],
        n_spatial_tokens=n_tokens,
        n_proprio_tokens=config['policy'].get('n_context_tokens', 4),
        action_dim=config['policy']['action_dim'],
        chunk_size=config['policy']['chunk_size'],
        device=device,
    )

    # Print parameter counts
    counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count / 1e6:.2f}M")

    # Get embedding settings
    noise_std = config.get('robust_embeddings', {}).get('noise_std', 0.05)
    normalize_emb = config.get('robust_embeddings', {}).get('normalize', True)

    # Create datasets with gripper-focused augmentation
    print("\n" + "=" * 60)
    print("GRIPPER-FOCUSED DATA AUGMENTATION")
    print("=" * 60)
    print(f"  Oversample factor: {args.oversample_factor}×")
    print(f"  Jitter range: ±{args.jitter_range} steps")
    print(f"  Transition weight: {args.transition_weight}×")
    print(f"  Gripper loss weight: {args.gripper_loss_weight}×")
    print(f"  Balance gripper classes: {args.balance_gripper}")
    print("=" * 60 + "\n")

    train_dataset = GripperFocusedSpatialDataset(
        embedding_dir=embedding_dir,
        split='train',
        train_ratio=config['data'].get('train_ratio', 0.9),
        seed=config['training'].get('seed', 42),
        # Gripper augmentation
        gripper_augmentation=True,
        oversample_transitions=True,
        oversample_factor=args.oversample_factor,
        jitter_range=args.jitter_range,
        jitter_prob=0.5,
        transition_weight=args.transition_weight,
        transition_window=10,
        # Embedding augmentation
        noise_std=noise_std,
        normalize=normalize_emb,
    )

    val_dataset = GripperFocusedSpatialDataset(
        embedding_dir=embedding_dir,
        split='val',
        train_ratio=config['data'].get('train_ratio', 0.9),
        seed=config['training'].get('seed', 42),
        # No augmentation for validation
        gripper_augmentation=False,
        oversample_transitions=False,
        noise_std=0.0,  # No noise
        normalize=normalize_emb,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer with gripper-balanced loss
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_epochs=config['training']['warmup_epochs'],
        grad_clip=config['training']['grad_clip'],
        device=device,
        log_every=config['logging']['log_every'],
        save_dir=args.save_dir,
        use_wandb=use_wandb,
        use_precomputed=True,
        seed=config['training'].get('seed', 42),
        # Loss configuration - use gripper_balanced loss
        loss_type="gripper_balanced",
        start_weight=config.get('loss', {}).get('start_weight', 3.0),
        start_timesteps=config.get('loss', {}).get('start_timesteps', 10),
        gripper_transition_weight=args.transition_weight,
        gripper_dim=6,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train(n_epochs=config['training']['epochs'])

    if use_wandb:
        wandb.finish()

    print("\nTraining complete!")
    print(f"Best model saved to: {args.save_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train with gripper-focused data augmentation"
    )

    # Basic args
    parser.add_argument("--config", type=str, default="configs/spatial.yaml",
                        help="Path to config file")
    parser.add_argument("--suite", type=str, default=None,
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10'],
                        help="Override LIBERO suite")
    parser.add_argument("--embedding_dir", type=str, default="/workspace/data/embeddings",
                        help="Directory with pre-computed embeddings")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--save_dir", type=str, default="/workspace/checkpoints_gripper_v1",
                        help="Directory to save checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    # Gripper augmentation args
    parser.add_argument("--oversample_factor", type=int, default=5,
                        help="How many times to oversample transitions (default: 5)")
    parser.add_argument("--jitter_range", type=int, default=5,
                        help="Max timesteps to shift transitions (default: ±5)")
    parser.add_argument("--transition_weight", type=float, default=5.0,
                        help="Loss weight for transition timesteps (default: 5.0)")
    parser.add_argument("--gripper_loss_weight", type=float, default=2.0,
                        help="Extra weight for gripper dimension (default: 2.0)")
    parser.add_argument("--balance_gripper", type=bool, default=True,
                        help="Balance OPEN vs CLOSE samples (default: True)")

    args = parser.parse_args()
    main(args)
