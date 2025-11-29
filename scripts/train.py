"""
Main training script for V-JEPA 2 Policy
"""

import argparse
import yaml
import wandb
import torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.full_model import VJEPA2Policy
from src.data.libero_dataset import create_dataloader
from src.training.trainer import Trainer


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
    if not args.no_wandb:
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=args.run_name,
        )
    
    # Create model
    print("Creating model...")
    model = VJEPA2Policy(
        vjepa2_model=config['vjepa2']['model_name'],
        vjepa2_freeze=config['vjepa2']['freeze'],
        proprio_dim=config['proprio']['dim'],
        proprio_history=config['proprio']['history_len'],
        policy_hidden_dim=config['policy']['hidden_dim'],
        policy_n_heads=config['policy']['n_heads'],
        policy_n_layers=config['policy']['n_layers'],
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
    )
    
    val_loader = create_dataloader(
        data_dir=config['data']['libero_path'],
        suite=config['data']['suite'],
        batch_size=config['training']['batch_size'],
        split='val',
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
        save_dir=args.save_dir,
        use_wandb=not args.no_wandb,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(n_epochs=config['training']['epochs'])
    
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
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    args = parser.parse_args()
    main(args)