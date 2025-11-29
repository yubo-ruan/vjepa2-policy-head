"""
Pre-compute V-JEPA 2 embeddings for faster training.

This script processes all samples once through V-JEPA 2 and saves the embeddings.
Training then loads these pre-computed embeddings instead of running the encoder.

Speedup: ~30x (6 sec/batch -> 0.2 sec/batch)
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.vjepa2_encoder import VJEPA2Encoder
from vjepa_policy.data.libero_dataset import LIBERODataset


def precompute_embeddings(
    data_dir: str,
    suite: str,
    output_dir: str,
    model_path: str = "/workspace/models/vjepa2-ac-vitg.pt",
    model_name: str = "vjepa2_vitg",
    batch_size: int = 8,
    split: str = "train",
    train_ratio: float = 0.9,
    seed: int = 42,
):
    """Pre-compute V-JEPA 2 embeddings for a dataset split."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory
    output_path = Path(output_dir) / suite / split
    output_path.mkdir(parents=True, exist_ok=True)

    # Load encoder
    # IMPORTANT: Use mean pooling for reproducibility (AttentivePooler is randomly initialized)
    print(f"Loading V-JEPA 2 encoder with mean pooling...")
    encoder = VJEPA2Encoder(
        model_path=model_path,
        model_name=model_name,
        freeze=True,
        use_attentive_pool=False,  # Mean pooling for reproducible embeddings
        device=device,
    )
    encoder.eval()

    # Load dataset
    print(f"Loading {split} dataset...")
    dataset = LIBERODataset(
        data_dir=data_dir,
        suite=suite,
        split=split,
        train_ratio=train_ratio,
        video_len=16,
        proprio_history=5,
        chunk_size=50,
        image_size=256,
        seed=seed,
    )

    print(f"Processing {len(dataset)} samples...")
    print(f"Output directory: {output_path}")

    # Process in batches for efficiency
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    sample_idx = 0
    for batch_idx in tqdm(range(n_batches), desc=f"Computing embeddings ({split})"):
        # Collect batch
        batch_samples = []
        batch_indices = []

        for i in range(batch_size):
            idx = batch_idx * batch_size + i
            if idx >= len(dataset):
                break
            batch_samples.append(dataset[idx])
            batch_indices.append(idx)

        if len(batch_samples) == 0:
            break

        # Stack tensors
        video = torch.stack([s['video'] for s in batch_samples]).to(device)
        goal = torch.stack([s['goal'] for s in batch_samples]).to(device)
        proprio = torch.stack([s['proprio'] for s in batch_samples])
        actions = torch.stack([s['actions'] for s in batch_samples])

        # Compute embeddings
        with torch.no_grad():
            video_emb = encoder.encode_video(video)  # (B, embed_dim)
            goal_emb = encoder.encode_image(goal)     # (B, embed_dim)

        # Save each sample individually
        for i, idx in enumerate(batch_indices):
            sample_data = {
                'video_emb': video_emb[i].cpu(),
                'goal_emb': goal_emb[i].cpu(),
                'proprio': proprio[i],
                'actions': actions[i],
                'demo_idx': batch_samples[i]['demo_idx'],
                'start_frame': batch_samples[i]['start_frame'],
            }

            save_path = output_path / f"sample_{idx:06d}.pt"
            torch.save(sample_data, save_path)
            sample_idx += 1

    print(f"\nSaved {sample_idx} embeddings to {output_path}")
    return sample_idx


def main():
    parser = argparse.ArgumentParser(description="Pre-compute V-JEPA 2 embeddings")

    parser.add_argument("--data_dir", type=str, default="/workspace/data/libero",
                        help="Path to LIBERO data")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10'],
                        help="LIBERO suite")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--model_path", type=str, default="/workspace/models/vjepa2-ac-vitg.pt",
                        help="Path to V-JEPA 2 checkpoint")
    parser.add_argument("--model_name", type=str, default="vjepa2_vitg",
                        help="Model variant")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Process both splits
    print("=" * 60)
    print("Pre-computing V-JEPA 2 Embeddings")
    print("=" * 60)

    total_samples = 0

    for split in ['train', 'val']:
        print(f"\n{'=' * 40}")
        print(f"Processing {split} split...")
        print(f"{'=' * 40}")

        n_samples = precompute_embeddings(
            data_dir=args.data_dir,
            suite=args.suite,
            output_dir=args.output_dir,
            model_path=args.model_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            split=split,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        total_samples += n_samples

    print(f"\n{'=' * 60}")
    print(f"Done! Total samples processed: {total_samples}")
    print(f"Embeddings saved to: {args.output_dir}/{args.suite}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
