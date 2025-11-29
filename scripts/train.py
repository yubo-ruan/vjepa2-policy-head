"""
Main training script for V-JEPA 2 Policy
"""

import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2Policy
from vjepa_policy.data.libero_dataset import create_dataloader
from vjepa_policy.training.trainer import Trainer

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
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=args.run_name,
        )

    # Create model
    print("Creating model...")
    model = VJEPA2Policy(
        vjepa2_model_path=config['vjepa2']['model_path'],
        vjepa2_model_name=config['vjepa2']['model_name'],
        vjepa2_freeze=config['vjepa2']['freeze'],
        vjepa2_num_frames=config['vjepa2']['num_frames'],
        vjepa2_use_attentive_pool=config['vjepa2'].get('use_attentive_pool', True),
        proprio_dim=config['proprio']['dim'],
        proprio_history=config['proprio']['history_len'],
        proprio_output_dim=config['proprio'].get('output_dim', 256),
        policy_hidden_dim=config['policy']['hidden_dim'],
        policy_n_heads=config['policy']['n_heads'],
        policy_n_layers=config['policy']['n_layers'],
        policy_n_context_tokens=config['policy'].get('n_context_tokens', 4),
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
    print("\nLoading data...")
    train_loader = create_dataloader(
        data_dir=config['data']['libero_path'],
        suite=config['data']['suite'],
        batch_size=config['training']['batch_size'],
        split='train',
        train_ratio=config['data'].get('train_ratio', 0.9),
        sample_stride=config['data'].get('sample_stride', 5),
        seed=config['training'].get('seed', 42),
    )

    val_loader = create_dataloader(
        data_dir=config['data']['libero_path'],
        suite=config['data']['suite'],
        batch_size=config['training']['batch_size'],
        split='val',
        train_ratio=config['data'].get('train_ratio', 0.9),
        sample_stride=config['data'].get('sample_stride', 5),
        seed=config['training'].get('seed', 42),
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
        seed=config['training'].get('seed', 42),
    )

    # Train
    trainer.train(n_epochs=config['training']['epochs'])

    if use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V-JEPA 2 Policy")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
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
