#!/usr/bin/env python3
"""
Unified Precompute Script for V-JEPA 2 Spatial Embeddings.

This script precomputes spatial token embeddings from LIBERO demonstrations,
enabling fast training without running the V-JEPA 2 encoder during each batch.

Why Precompute?
    - V-JEPA 2 ViT-Giant has ~1.1B parameters and is slow to run
    - Training with live encoding would be bottlenecked by encoder forward pass
    - Precomputing once and loading embeddings is ~10x faster for training
    - Enables rapid iteration on policy head architecture

Output Format:
    Each sample is saved as a .pt file containing:
        - video_tokens: (64, 1408) spatial tokens from 16-frame video
        - goal_tokens: (64, 1408) spatial tokens from goal image
        - proprio: (75,) flattened proprioception history (5 * 15)
        - actions: (50, 7) action chunk

Static Frame Augmentation:
    During evaluation, the robot starts from a static pose. During training,
    we sample from the middle of trajectories where there's motion. This creates
    a distribution shift that hurts evaluation performance.

    Static augmentation addresses this by:
    - static_prob: Repeat first frame for entire video (simulates static start)
    - beginning_prob: Sample from beginning of episode (near-static states)

    Recommended: --static_aug (uses 25% static + 25% beginning-biased)

Usage:
    # Basic precomputation
    python scripts/precompute.py --suite libero_spatial --output_dir /workspace/data/embeddings

    # With static frame augmentation (recommended for better generalization)
    python scripts/precompute.py --suite libero_spatial --output_dir /workspace/data/embeddings --static_aug

    # Custom augmentation probabilities
    python scripts/precompute.py --suite libero_spatial --output_dir /workspace/data/embeddings \
        --static_aug --static_prob 0.3 --beginning_prob 0.2

CLI Arguments:
    --suite: LIBERO suite to process (required)
    --output_dir: Directory to save embeddings (required)
    --libero_path: Path to LIBERO data directory
    --model_path: Path to V-JEPA 2 weights
    --train_ratio: Train/val split ratio (default: 0.9)
    --batch_size: Batch size for encoding (default: 8)
    --seed: Random seed for reproducibility
    --static_aug: Enable static frame augmentation
    --static_prob: Probability of fully static video (default: 0.25)
    --beginning_prob: Probability of beginning-biased sampling (default: 0.25)
    --device: Device to run on (cuda/cpu)
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import sys
import random
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.vjepa2_encoder import VJEPA2Encoder
from vjepa_policy.data.libero_dataset import LIBERODataset


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AugmentedLIBERODataset:
    """
    LIBERO dataset wrapper with optional static frame augmentation.

    This wrapper addresses the train/eval distribution shift problem:
    - Training data: Sampled from middle of trajectories (lots of motion)
    - Evaluation: Robot starts from static initial pose

    The augmentation strategy:
    - static_prob: Fully static video (first frame repeated 16 times)
      This simulates what the model sees at the very start of evaluation.

    - beginning_prob: Sample from beginning of episode (first 30 frames)
      This provides near-static states with minimal motion.

    - Remaining samples: Normal random sampling from demonstration

    Args:
        base_dataset: LIBERODataset instance to wrap
        static_prob: Probability of static video augmentation (default: 0.0)
        beginning_prob: Probability of beginning-biased sampling (default: 0.0)
        max_beginning_frame: Max frame for beginning-biased sampling (default: 30)
    """

    def __init__(
        self,
        base_dataset: LIBERODataset,
        static_prob: float = 0.0,
        beginning_prob: float = 0.0,
        max_beginning_frame: int = 30,
    ):
        self.base_dataset = base_dataset
        self.static_prob = static_prob
        self.beginning_prob = beginning_prob
        self.max_beginning_frame = max_beginning_frame

        # Store original samples (demo_idx, start_frame pairs)
        self.samples = list(base_dataset.samples)

        # Pre-assign augmentation type to each sample for reproducibility
        self.aug_types = self._create_aug_types()

        if static_prob > 0 or beginning_prob > 0:
            print(f"Augmentation: static={static_prob:.0%}, beginning={beginning_prob:.0%}")

    def _create_aug_types(self):
        """
        Assign augmentation type to each sample.

        Returns:
            List of augmentation types ('static', 'beginning', or 'normal')
        """
        aug_types = []
        for _ in self.samples:
            r = random.random()
            if r < self.static_prob:
                aug_types.append('static')
            elif r < self.static_prob + self.beginning_prob:
                aug_types.append('beginning')
            else:
                aug_types.append('normal')
        return aug_types

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample with optional augmentation applied.

        For static augmentation: First frame is repeated for entire video
        For beginning augmentation: Video is sampled from first 30 frames
        For normal: Original random sampling is used

        Args:
            idx: Sample index

        Returns:
            Dictionary with video, goal, proprio, actions, aug_type
        """
        aug_type = self.aug_types[idx]
        demo_idx, start_frame = self.samples[idx]

        # Load demonstration data
        demo = self.base_dataset.demos[demo_idx]
        demo_data = self.base_dataset._load_demo_data(demo)

        # Get dataset parameters
        video_len = self.base_dataset.video_len
        chunk_size = self.base_dataset.chunk_size

        # Determine video start frame based on augmentation type
        if aug_type == 'static':
            # Use first frame repeated - simulates static start of evaluation
            video_start = 0
        elif aug_type == 'beginning':
            # Sample from beginning of episode - near-static states
            max_start = min(self.max_beginning_frame, len(demo_data['images']) - video_len - chunk_size)
            video_start = random.randint(0, max(0, max_start))
        else:
            # Normal: use original randomly sampled start frame
            video_start = start_frame

        # Extract video frames
        video_end = video_start + video_len
        if aug_type == 'static':
            # Repeat first frame for entire video (key augmentation!)
            video = np.stack([demo_data['images'][0]] * video_len, axis=0)
        else:
            video = demo_data['images'][video_start:video_end]

        # Get goal image (last frame of demonstration)
        goal = demo_data['images'][-1]

        # Determine action prediction point (last frame of video)
        action_point = video_start + video_len - 1

        # Extract proprioception history ending at action point
        proprio_history = self.base_dataset.proprio_history
        proprio_start = max(0, action_point - proprio_history + 1)
        proprio_end = action_point + 1
        proprio = demo_data['proprio'][proprio_start:proprio_end]

        # Pad proprio if at beginning of episode
        if len(proprio) < proprio_history:
            pad_len = proprio_history - len(proprio)
            pad = np.repeat(proprio[:1], pad_len, axis=0)
            proprio = np.concatenate([pad, proprio], axis=0)

        # Extract action chunk starting at action point
        action_end = action_point + chunk_size
        actions = demo_data['actions'][action_point:action_end]

        # Pad actions if near end of episode
        if len(actions) < chunk_size:
            pad_len = chunk_size - len(actions)
            pad = np.repeat(actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        # Convert to tensors with proper format
        # Video: (T, H, W, C) -> (T, C, H, W), normalized to [0, 1]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        # Goal: (H, W, C) -> (C, H, W), normalized to [0, 1]
        goal = torch.from_numpy(goal).permute(2, 0, 1).float() / 255.0
        proprio = torch.from_numpy(proprio.copy()).float()
        actions = torch.from_numpy(actions.copy()).float()

        # Resize if needed (V-JEPA 2 expects 256x256)
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
        }


def main():
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(description='Precompute V-JEPA 2 Spatial Embeddings')

    # Required arguments
    parser.add_argument('--suite', type=str, required=True,
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10'],
                        help='LIBERO suite to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for embeddings')

    # Data paths
    parser.add_argument('--libero_path', type=str, default='/workspace/data/libero',
                        help='Path to LIBERO data')
    parser.add_argument('--model_path', type=str, default='/workspace/models/vjepa2-ac-vitg.pt',
                        help='Path to V-JEPA 2 weights')

    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Train/val split ratio')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for encoding')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Augmentation options (key for generalization!)
    parser.add_argument('--static_aug', action='store_true',
                        help='Enable static frame augmentation')
    parser.add_argument('--static_prob', type=float, default=0.25,
                        help='Probability of static video augmentation')
    parser.add_argument('--beginning_prob', type=float, default=0.25,
                        help='Probability of beginning-biased sampling')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # ========== Setup ==========
    set_seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir) / f"{args.suite}_spatial"
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 60)
    print("V-JEPA 2 Spatial Embedding Precomputation")
    print("=" * 60)
    print(f"Suite: {args.suite}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Static augmentation: {args.static_aug}")
    if args.static_aug:
        print(f"  Static prob: {args.static_prob}")
        print(f"  Beginning prob: {args.beginning_prob}")
    print("=" * 60)
    print()

    # ========== Load V-JEPA 2 Encoder ==========
    print("Loading V-JEPA 2 encoder...")
    encoder = VJEPA2Encoder(
        model_path=args.model_path,
        model_name='vjepa2_vitg',
        freeze=True,  # Always frozen for precomputation
        device=args.device,
        num_frames=16,
        use_attentive_pool=False,  # We use spatial encoding (64 tokens)
    )
    encoder.eval()
    print()

    # ========== Create Datasets ==========
    print("Loading LIBERO data...")

    # Create base datasets with train/val split
    train_dataset_base = LIBERODataset(
        data_dir=args.libero_path,
        suite=args.suite,
        split='train',
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    val_dataset_base = LIBERODataset(
        data_dir=args.libero_path,
        suite=args.suite,
        split='val',
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Wrap with augmentation
    # Only apply augmentation to training set
    if args.static_aug:
        train_dataset = AugmentedLIBERODataset(
            train_dataset_base,
            static_prob=args.static_prob,
            beginning_prob=args.beginning_prob,
        )
        # No augmentation for validation - we want clean evaluation
        val_dataset = AugmentedLIBERODataset(val_dataset_base, static_prob=0, beginning_prob=0)
    else:
        train_dataset = AugmentedLIBERODataset(train_dataset_base, static_prob=0, beginning_prob=0)
        val_dataset = AugmentedLIBERODataset(val_dataset_base, static_prob=0, beginning_prob=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # ========== Process Function ==========
    def process_dataset(dataset, output_path, desc):
        """
        Process dataset and save spatial embeddings.

        For each sample:
        1. Load video and goal image
        2. Run V-JEPA 2 encoder to get spatial tokens
        3. Save tokens along with proprio and actions

        Args:
            dataset: AugmentedLIBERODataset instance
            output_path: Directory to save .pt files
            desc: Description for progress bar
        """
        print(f"\nProcessing {desc}...")

        for idx in tqdm(range(len(dataset)), desc=desc):
            sample = dataset[idx]

            # Prepare video: (T, C, H, W) -> (1, C, T, H, W) for V-JEPA 2
            video = sample['video'].unsqueeze(0).permute(0, 2, 1, 3, 4).to(args.device)

            # Prepare goal: (C, H, W) -> (1, C, H, W)
            goal = sample['goal'].unsqueeze(0).to(args.device)

            # Encode to spatial tokens (64 per modality)
            with torch.no_grad():
                video_tokens = encoder.encode_video_spatial(video)  # (1, 64, 1408)
                goal_tokens = encoder.encode_image_spatial(goal)    # (1, 64, 1408)

            # Flatten proprio history: (5, 15) -> (75,)
            proprio = sample['proprio']
            if proprio.dim() == 2:
                proprio = proprio.flatten()

            # Save with consistent naming convention
            output_data = {
                'video_tokens': video_tokens.squeeze(0).cpu(),  # (64, 1408)
                'goal_tokens': goal_tokens.squeeze(0).cpu(),    # (64, 1408)
                'proprio': proprio.cpu(),                        # (75,)
                'actions': sample['actions'].cpu(),              # (50, 7)
            }

            torch.save(output_data, output_path / f'sample_{idx:06d}.pt')

    # ========== Process Train and Val ==========
    process_dataset(train_dataset, train_dir, "Train")
    process_dataset(val_dataset, val_dir, "Val")

    # ========== Done ==========
    print()
    print("=" * 60)
    print("Precomputation Complete!")
    print(f"Train embeddings: {train_dir}")
    print(f"Val embeddings: {val_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
