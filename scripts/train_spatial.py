"""
Training script for V-JEPA 2 Policy with SPATIAL tokens.

Uses pre-computed spatial embeddings (64, 1408) instead of mean-pooled (1408,).
This preserves spatial structure for better policy learning.

Usage:
1. First run: python scripts/precompute_spatial_embeddings.py --suite libero_spatial
2. Then run: python scripts/train_spatial.py --suite libero_spatial

Key differences from train_fast.py:
- Uses VJEPA2PolicySpatial model (with PolicyHeadSpatial)
- Loads spatial tokens (64, 1408) instead of pooled embeddings
- 132 context tokens: 64 video + 64 goal + 4 proprio
"""

import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2PolicySpatial
from vjepa_policy.data.libero_dataset import (
    PrecomputedSpatialEmbeddingDataset,
    RobustSpatialEmbeddingDataset,
)
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

    # Check embeddings exist (spatial directory with _spatial suffix)
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
            config=config,
            name=args.run_name or f"{suite}_spatial",
        )

    # Get embedding dimension from a sample
    sample_path = list((embedding_dir / "train").glob("*.pt"))[0]
    sample_data = torch.load(sample_path)
    n_tokens, embed_dim = sample_data['video_emb'].shape
    print(f"Detected spatial tokens: ({n_tokens}, {embed_dim})")

    # Create spatial model
    print("Creating VJEPA2PolicySpatial model...")
    model = VJEPA2PolicySpatial(
        vjepa2_model_path=config['vjepa2']['model_path'],
        vjepa2_model_name=config['vjepa2']['model_name'],
        vjepa2_freeze=True,  # Always frozen with precomputed
        vjepa2_num_frames=config['vjepa2']['num_frames'],
        proprio_dim=config['proprio']['dim'],
        proprio_history=config['proprio']['history_len'],
        proprio_output_dim=config['proprio'].get('output_dim', 256),
        policy_hidden_dim=config['policy']['hidden_dim'],
        policy_n_heads=config['policy']['n_heads'],
        policy_n_layers=config['policy']['n_layers'],
        n_spatial_tokens=n_tokens,  # 64
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

    # Create dataloaders
    print("\nLoading pre-computed spatial embeddings...")

    # Get robust embedding settings from config or use defaults
    noise_std = config.get('robust_embeddings', {}).get('noise_std', 0.05)
    normalize_emb = config.get('robust_embeddings', {}).get('normalize', True)
    use_robust = config.get('robust_embeddings', {}).get('enabled', True)

    if use_robust:
        print(f"Using RobustSpatialEmbeddingDataset: noise_std={noise_std}, normalize={normalize_emb}")

    base_train_dataset = PrecomputedSpatialEmbeddingDataset(
        embedding_dir=embedding_dir,
        split='train',
        train_ratio=config['data'].get('train_ratio', 0.9),
        seed=config['training'].get('seed', 42),
    )

    base_val_dataset = PrecomputedSpatialEmbeddingDataset(
        embedding_dir=embedding_dir,
        split='val',
        train_ratio=config['data'].get('train_ratio', 0.9),
        seed=config['training'].get('seed', 42),
    )

    # Wrap with RobustSpatialEmbeddingDataset for noise and normalization
    if use_robust:
        train_dataset = RobustSpatialEmbeddingDataset(
            base_train_dataset,
            noise_std=noise_std,
            normalize=normalize_emb,
            training=True,
        )
        val_dataset = RobustSpatialEmbeddingDataset(
            base_val_dataset,
            noise_std=noise_std,
            normalize=normalize_emb,
            training=False,  # No noise during validation
        )
    else:
        train_dataset = base_train_dataset
        val_dataset = base_val_dataset

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

    # Create trainer
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
        save_dir=args.save_dir or config['logging']['checkpoint_dir'],
        use_wandb=use_wandb,
        use_precomputed=True,  # Using precomputed spatial embeddings
        seed=config['training'].get('seed', 42),
    )

    # Train
    trainer.train(n_epochs=config['training']['epochs'])

    if use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with pre-computed spatial embeddings")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--suite", type=str, default=None,
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10'],
                        help="Override LIBERO suite")
    parser.add_argument("--embedding_dir", type=str, default="/workspace/data/embeddings",
                        help="Directory with pre-computed spatial embeddings")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    args = parser.parse_args()
    main(args)
