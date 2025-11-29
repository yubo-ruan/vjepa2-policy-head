"""
Pre-compute V-JEPA 2 SPATIAL embeddings for faster training.

This script:
1. Generates spatial tokens (64, 1408) instead of mean-pooled (1408,)
2. Applies static frame augmentation to handle beginning-of-episode distribution

Static Frame Augmentation:
- 30% chance: sample video window from first 30 frames (beginning-heavy)
- 20% chance: repeat the first frame for entire video (fully static)
- 50% chance: normal random sampling

This addresses the distribution shift where evaluation always starts with
static frames (16 identical frames) but training never sees this.

Speedup: ~30x (6 sec/batch -> 0.2 sec/batch)
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import random
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.vjepa2_encoder import VJEPA2Encoder
from vjepa_policy.data.libero_dataset import LIBERODataset


class StaticFrameAugmentedDataset:
    """
    Wrapper that applies static frame augmentation to a base dataset.

    Augmentation strategy (50% total):
    - 25% chance: fully static video (first frame repeated)
    - 25% chance: bias to beginning of episode (first 30 frames)
    - 50% chance: normal sampling

    This addresses high position error at trajectory start (0.5 vs 0.15 tolerance)
    by ensuring training sees more static/beginning frames like evaluation does.
    """

    def __init__(
        self,
        base_dataset: LIBERODataset,
        beginning_prob: float = 0.25,
        static_prob: float = 0.25,
        max_beginning_frame: int = 30,
    ):
        self.base_dataset = base_dataset
        self.beginning_prob = beginning_prob
        self.static_prob = static_prob
        self.max_beginning_frame = max_beginning_frame

        # Store original samples
        self.original_samples = list(base_dataset.samples)

        # Create augmented sample indices
        self.augmented_indices = self._create_augmented_indices()

        print(f"StaticFrameAugmentation: beginning={beginning_prob:.0%}, static={static_prob:.0%}")
        print(f"  Original samples: {len(self.original_samples)}")
        print(f"  Augmented samples: {len(self.augmented_indices)}")

    def _create_augmented_indices(self):
        """Create indices with augmentation type for each sample."""
        augmented = []

        for i, (demo_idx, start_frame) in enumerate(self.original_samples):
            # Sample augmentation type
            r = random.random()

            if r < self.static_prob:
                # Fully static: repeat first frame
                aug_type = 'static'
            elif r < self.static_prob + self.beginning_prob:
                # Beginning-biased: sample from early in episode
                aug_type = 'beginning'
            else:
                # Normal: use original sampling
                aug_type = 'normal'

            augmented.append({
                'original_idx': i,
                'demo_idx': demo_idx,
                'start_frame': start_frame,
                'aug_type': aug_type,
            })

        return augmented

    def __len__(self):
        return len(self.augmented_indices)

    def __getitem__(self, idx):
        aug_info = self.augmented_indices[idx]
        demo_idx = aug_info['demo_idx']
        start_frame = aug_info['start_frame']
        aug_type = aug_info['aug_type']

        demo = self.base_dataset.demos[demo_idx]
        demo_data = self.base_dataset._load_demo_data(demo)

        video_len = self.base_dataset.video_len
        chunk_size = self.base_dataset.chunk_size
        proprio_history = self.base_dataset.proprio_history

        # Apply augmentation
        if aug_type == 'static':
            # Use first frame repeated for entire video
            # But keep original start_frame for action/proprio alignment
            video_frames = np.stack([demo_data['images'][0]] * video_len, axis=0)
            action_point = start_frame + video_len - 1
        elif aug_type == 'beginning':
            # Bias sampling to beginning of episode
            max_start = min(self.max_beginning_frame, demo['length'] - video_len - chunk_size)
            if max_start > 0:
                new_start = random.randint(0, max_start)
                start_frame = new_start
            video_frames = demo_data['images'][start_frame:start_frame + video_len]
            action_point = start_frame + video_len - 1
        else:  # normal
            video_frames = demo_data['images'][start_frame:start_frame + video_len]
            action_point = start_frame + video_len - 1

        # Extract goal (last frame of demo)
        goal = demo_data['images'][-1]

        # Extract proprio history
        proprio_start = max(0, action_point - proprio_history + 1)
        proprio_end = action_point + 1
        proprio = demo_data['proprio'][proprio_start:proprio_end]

        # Pad proprio if needed
        if len(proprio) < proprio_history:
            pad_len = proprio_history - len(proprio)
            pad = np.repeat(proprio[:1], pad_len, axis=0)
            proprio = np.concatenate([pad, proprio], axis=0)

        # Extract action chunk
        action_end = min(action_point + chunk_size, len(demo_data['actions']))
        actions = demo_data['actions'][action_point:action_end]

        # Pad actions if needed
        if len(actions) < chunk_size:
            pad_len = chunk_size - len(actions)
            pad = np.repeat(actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        # Convert to tensors
        import torch.nn.functional as F
        video = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() / 255.0
        goal = torch.from_numpy(goal).permute(2, 0, 1).float() / 255.0
        proprio = torch.from_numpy(proprio.copy()).float()
        actions = torch.from_numpy(actions.copy()).float()

        # Resize if needed
        image_size = self.base_dataset.image_size
        if video.shape[-1] != image_size or video.shape[-2] != image_size:
            video = F.interpolate(video, size=(image_size, image_size), mode='bilinear', align_corners=False)
            goal = F.interpolate(goal.unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False).squeeze(0)

        return {
            'video': video,
            'goal': goal,
            'proprio': proprio,
            'actions': actions,
            'aug_type': aug_type,
            'demo_idx': demo_idx,
            'start_frame': start_frame,
        }


def precompute_spatial_embeddings(
    data_dir: str,
    suite: str,
    output_dir: str,
    model_path: str = "/workspace/models/vjepa2-ac-vitg.pt",
    model_name: str = "vjepa2_vitg",
    batch_size: int = 8,
    split: str = "train",
    train_ratio: float = 0.9,
    seed: int = 42,
    use_augmentation: bool = True,
    static_prob: float = 0.25,
    beginning_prob: float = 0.25,
):
    """Pre-compute V-JEPA 2 spatial embeddings for a dataset split."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory with _spatial suffix
    output_path = Path(output_dir) / f"{suite}_spatial" / split
    output_path.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"Loading V-JEPA 2 encoder (spatial mode)...")
    encoder = VJEPA2Encoder(
        model_path=model_path,
        model_name=model_name,
        freeze=True,
        use_attentive_pool=False,  # We use spatial encoding instead
        device=device,
    )
    encoder.eval()

    # Load base dataset
    print(f"Loading {split} dataset...")
    base_dataset = LIBERODataset(
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

    # Apply augmentation for training split only
    if use_augmentation and split == 'train':
        dataset = StaticFrameAugmentedDataset(
            base_dataset,
            static_prob=static_prob,
            beginning_prob=beginning_prob,
        )
    else:
        dataset = base_dataset

    print(f"Processing {len(dataset)} samples...")
    print(f"Output directory: {output_path}")

    # Process in batches
    n_batches = (len(dataset) + batch_size - 1) // batch_size

    sample_idx = 0
    aug_counts = {'static': 0, 'beginning': 0, 'normal': 0}

    for batch_idx in tqdm(range(n_batches), desc=f"Computing spatial embeddings ({split})"):
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

        # Compute spatial embeddings: (B, 64, 1408)
        with torch.no_grad():
            video_emb = encoder.encode_video_spatial(video)  # (B, 64, 1408)
            goal_emb = encoder.encode_image_spatial(goal)     # (B, 64, 1408)

        # Save each sample individually
        for i, idx in enumerate(batch_indices):
            aug_type = batch_samples[i].get('aug_type', 'normal')
            aug_counts[aug_type] += 1

            sample_data = {
                'video_emb': video_emb[i].cpu(),  # (64, 1408)
                'goal_emb': goal_emb[i].cpu(),     # (64, 1408)
                'proprio': proprio[i],
                'actions': actions[i],
                'aug_type': aug_type,
                'demo_idx': batch_samples[i]['demo_idx'],
                'start_frame': batch_samples[i]['start_frame'],
            }

            save_path = output_path / f"sample_{sample_idx:06d}.pt"
            torch.save(sample_data, save_path)
            sample_idx += 1

    print(f"\nSaved {sample_idx} spatial embeddings to {output_path}")
    if use_augmentation and split == 'train':
        print(f"Augmentation counts: {aug_counts}")
    return sample_idx


def main():
    parser = argparse.ArgumentParser(description="Pre-compute V-JEPA 2 spatial embeddings")

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
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable static frame augmentation")
    parser.add_argument("--static_prob", type=float, default=0.25,
                        help="Probability of fully static video augmentation (default: 0.25)")
    parser.add_argument("--beginning_prob", type=float, default=0.25,
                        help="Probability of beginning-biased sampling (default: 0.25)")

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Process both splits
    print("=" * 60)
    print("Pre-computing V-JEPA 2 Spatial Embeddings")
    print("=" * 60)

    total_samples = 0

    for split in ['train', 'val']:
        print(f"\n{'=' * 40}")
        print(f"Processing {split} split...")
        print(f"{'=' * 40}")

        n_samples = precompute_spatial_embeddings(
            data_dir=args.data_dir,
            suite=args.suite,
            output_dir=args.output_dir,
            model_path=args.model_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            split=split,
            train_ratio=args.train_ratio,
            seed=args.seed,
            use_augmentation=not args.no_augmentation,
            static_prob=args.static_prob,
            beginning_prob=args.beginning_prob,
        )
        total_samples += n_samples

    print(f"\n{'=' * 60}")
    print(f"Done! Total samples processed: {total_samples}")
    print(f"Spatial embeddings saved to: {args.output_dir}/{args.suite}_spatial/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
