"""
LIBERO Dataset Loader

Loads demonstrations from LIBERO benchmark for training.
Supports pre-computed embeddings for faster training.
"""

import os
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random


class LIBERODataset(Dataset):
    """
    LIBERO demonstration dataset.

    Each sample contains:
    - video: 16 frames of observation
    - goal: final frame of demonstration
    - proprio: proprioception history (aligned with last video frame)
    - actions: action chunk (50 actions starting from last video frame)
    """

    def __init__(
        self,
        data_dir: str,
        suite: str = "libero_object",
        split: str = "train",
        train_ratio: float = 0.9,
        video_len: int = 16,
        proprio_history: int = 5,
        chunk_size: int = 50,
        image_size: int = 256,
        frame_skip: int = 1,
        sample_stride: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Path to LIBERO data
            suite: One of libero_object, libero_spatial, libero_goal, libero_90, libero_10
            split: 'train' or 'val'
            train_ratio: Ratio of demos for training (rest for validation)
            video_len: Number of frames for video input
            proprio_history: Number of proprio frames
            chunk_size: Number of actions to predict
            image_size: Image resolution
            frame_skip: Skip frames (1 = use every frame)
            sample_stride: Stride between samples in a demo
            seed: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.suite = suite
        self.split = split
        self.train_ratio = train_ratio
        self.video_len = video_len
        self.proprio_history = proprio_history
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.sample_stride = sample_stride

        # Load demo metadata
        all_demos = self._load_all_demos()

        # Split into train/val
        self.demos = self._split_demos(all_demos, seed)

        # Create index of valid starting points
        self.samples = self._create_sample_index()

        print(f"[{split}] Loaded {len(self.samples)} samples from {len(self.demos)} demos ({suite})")

    def _load_all_demos(self) -> List[Dict]:
        """Load all demonstrations for the suite"""
        demos = []

        suite_dir = self.data_dir / self.suite
        if not suite_dir.exists():
            raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

        demo_files = sorted(suite_dir.glob("*.hdf5"))

        if len(demo_files) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {suite_dir}")

        for demo_file in demo_files:
            try:
                with h5py.File(demo_file, 'r') as f:
                    # Get demo info
                    # Handle different LIBERO HDF5 structures
                    if 'data' in f:
                        # LIBERO format: data/demo_0, data/demo_1, etc.
                        for demo_key in f['data'].keys():
                            demo_grp = f['data'][demo_key]
                            demo_data = {
                                'path': str(demo_file),
                                'demo_key': demo_key,
                                'length': demo_grp['actions'].shape[0],
                                'task_name': demo_file.stem,
                            }
                            demos.append(demo_data)
                    else:
                        # Simple format: actions at root
                        demo_data = {
                            'path': str(demo_file),
                            'demo_key': None,
                            'length': f['actions'].shape[0],
                            'task_name': f.attrs.get('task_name', demo_file.stem),
                        }
                        demos.append(demo_data)
            except Exception as e:
                print(f"Warning: Could not load {demo_file}: {e}")

        return demos

    def _split_demos(self, all_demos: List[Dict], seed: int) -> List[Dict]:
        """Split demos into train/val"""
        random.seed(seed)

        # Shuffle demos
        shuffled = all_demos.copy()
        random.shuffle(shuffled)

        # Split
        n_train = int(len(shuffled) * self.train_ratio)

        if self.split == 'train':
            return shuffled[:n_train]
        else:
            return shuffled[n_train:]

    def _create_sample_index(self) -> List[Tuple[int, int]]:
        """Create index of (demo_idx, start_frame) pairs"""
        samples = []

        # Minimum demo length needed
        min_length = self.video_len + self.chunk_size

        for demo_idx, demo in enumerate(self.demos):
            demo_len = demo['length']

            if demo_len < min_length:
                continue

            # Valid starting frames: 0 to (demo_len - min_length)
            max_start = demo_len - min_length

            # Sample starting points with stride
            for start in range(0, max_start + 1, self.sample_stride):
                samples.append((demo_idx, start))

        return samples

    def _load_demo_data(self, demo: Dict) -> Dict:
        """Load demo data from HDF5 file"""
        with h5py.File(demo['path'], 'r') as f:
            if demo['demo_key'] is not None:
                # LIBERO format
                grp = f['data'][demo['demo_key']]
                obs_grp = grp['obs']
            else:
                # Simple format
                grp = f
                obs_grp = f['obs'] if 'obs' in f else f

            data = {
                'images': self._get_images(obs_grp),
                'actions': grp['actions'][:],
                'proprio': self._extract_proprio(obs_grp),
            }
        return data

    def _get_images(self, obs_grp) -> np.ndarray:
        """Get images from observation group"""
        # Try different possible keys
        image_keys = ['agentview_image', 'agentview_rgb', 'image', 'rgb']
        for key in image_keys:
            if key in obs_grp:
                return obs_grp[key][:]
        raise KeyError(f"No image key found. Available: {list(obs_grp.keys())}")

    def _extract_proprio(self, obs_grp) -> np.ndarray:
        """Extract proprioception from observation group"""
        # LIBERO HDF5 structure:
        # ee_pos: (T, 3) - end effector position
        # ee_ori: (T, 3) - end effector orientation (euler angles or axis-angle)
        # gripper_states: (T, 2) - gripper state
        # joint_states: (T, 7) - joint positions
        # Total: 3 + 3 + 2 + 7 = 15 dims

        # Also try robosuite-style keys
        proprio_keys_libero = [
            'ee_pos',           # (T, 3)
            'ee_ori',           # (T, 3)
            'gripper_states',   # (T, 2)
            'joint_states',     # (T, 7)
        ]

        proprio_keys_robosuite = [
            ('robot0_eef_pos', 3),
            ('robot0_eef_quat', 4),
            ('robot0_gripper_qpos', 2),
            ('robot0_joint_pos', 7),
            ('robot0_joint_vel', 7),
        ]

        proprio_parts = []

        # Try LIBERO keys first
        for key in proprio_keys_libero:
            if key in obs_grp:
                part = obs_grp[key][:]
                proprio_parts.append(part)

        # If no LIBERO keys found, try robosuite keys
        if len(proprio_parts) == 0:
            for key, expected_dim in proprio_keys_robosuite:
                if key in obs_grp:
                    part = obs_grp[key][:]
                    proprio_parts.append(part)

        if len(proprio_parts) == 0:
            # Fallback: try to find any proprio-like keys
            for key in obs_grp.keys():
                if 'proprio' in key.lower() or 'state' in key.lower():
                    if 'rgb' not in key.lower() and 'image' not in key.lower():
                        proprio_parts.append(obs_grp[key][:])
                        break

        if len(proprio_parts) == 0:
            raise KeyError(f"No proprio found. Available: {list(obs_grp.keys())}")

        return np.concatenate(proprio_parts, axis=-1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo_idx, start_frame = self.samples[idx]
        demo = self.demos[demo_idx]

        # Load demo data
        demo_data = self._load_demo_data(demo)

        # Extract video frames: [start_frame, start_frame + video_len)
        video_end = start_frame + self.video_len
        video = demo_data['images'][start_frame:video_end]  # (video_len, H, W, 3)

        # Extract goal (last frame of demo)
        goal = demo_data['images'][-1]  # (H, W, 3)

        # Action prediction point: last frame of video
        action_point = start_frame + self.video_len - 1

        # Extract proprio history ENDING at action_point (the observation frame)
        # proprio_history frames: [action_point - proprio_history + 1, action_point + 1)
        proprio_start = max(0, action_point - self.proprio_history + 1)
        proprio_end = action_point + 1
        proprio = demo_data['proprio'][proprio_start:proprio_end]

        # Pad proprio at the beginning if needed
        if len(proprio) < self.proprio_history:
            pad_len = self.proprio_history - len(proprio)
            pad = np.repeat(proprio[:1], pad_len, axis=0)
            proprio = np.concatenate([pad, proprio], axis=0)

        # Extract action chunk starting from action_point
        action_end = action_point + self.chunk_size
        actions = demo_data['actions'][action_point:action_end]

        # Pad actions at the end if needed
        if len(actions) < self.chunk_size:
            pad_len = self.chunk_size - len(actions)
            pad = np.repeat(actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        # Convert to tensors
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        goal = torch.from_numpy(goal).permute(2, 0, 1).float() / 255.0       # (C, H, W)
        proprio = torch.from_numpy(proprio.copy()).float()                    # (history, proprio_dim)
        actions = torch.from_numpy(actions.copy()).float()                    # (chunk, action_dim)

        # Resize images to target size if needed (LIBERO is 128x128, V-JEPA 2 needs 256x256)
        if video.shape[-1] != self.image_size or video.shape[-2] != self.image_size:
            video = F.interpolate(video, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            goal = F.interpolate(goal.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)

        return {
            'video': video,
            'goal': goal,
            'proprio': proprio,
            'actions': actions,
            'demo_idx': demo_idx,
            'start_frame': start_frame,
        }


class PrecomputedEmbeddingDataset(Dataset):
    """
    Dataset with pre-computed V-JEPA 2 embeddings.
    Much faster for training (no encoder forward pass needed).

    Expected directory structure:
        embedding_dir/
            train/
                sample_000000.pt
                sample_000001.pt
                ...
            val/
                sample_000000.pt
                ...
    """

    def __init__(
        self,
        embedding_dir: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.split = split

        # Check for split subdirectory (new format)
        split_dir = self.embedding_dir / split
        if split_dir.exists():
            # New format: embeddings are pre-split into train/val directories
            self.files = sorted(split_dir.glob("*.pt"))
        else:
            # Old format: all files in one directory, split manually
            all_files = sorted(self.embedding_dir.glob("*.pt"))
            random.seed(seed)
            shuffled = all_files.copy()
            random.shuffle(shuffled)
            n_train = int(len(shuffled) * train_ratio)

            if split == 'train':
                self.files = shuffled[:n_train]
            else:
                self.files = shuffled[n_train:]

        print(f"[{split}] Loaded {len(self.files)} pre-computed samples")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.files[idx])

        # Support both naming conventions
        current_emb = data.get('current_emb', data.get('video_emb'))
        goal_emb = data.get('goal_emb')

        return {
            'current_emb': current_emb,
            'goal_emb': goal_emb,
            'proprio': data['proprio'],
            'actions': data['actions'],
        }


def create_dataloader(
    data_dir: str,
    suite: str = "libero_object",
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create LIBERO dataloader"""

    dataset = LIBERODataset(data_dir=data_dir, suite=suite, split=split, **kwargs)

    shuffle = (split == 'train')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return loader


def test_libero_dataset():
    """Test LIBERO dataset loading"""
    import sys

    # This test requires actual LIBERO data
    data_dir = "/workspace/data/libero"

    if not Path(data_dir).exists():
        print(f"LIBERO data not found at {data_dir}")
        print("Skipping dataset test")
        return

    print("Testing LIBERODataset...")

    # Test train split
    train_dataset = LIBERODataset(
        data_dir=data_dir,
        suite="libero_object",
        split="train",
        video_len=16,
        proprio_history=5,
        chunk_size=50,
    )

    # Test val split
    val_dataset = LIBERODataset(
        data_dir=data_dir,
        suite="libero_object",
        split="val",
        video_len=16,
        proprio_history=5,
        chunk_size=50,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Test sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample contents:")
        print(f"  video: {sample['video'].shape}")  # (16, 3, H, W)
        print(f"  goal: {sample['goal'].shape}")    # (3, H, W)
        print(f"  proprio: {sample['proprio'].shape}")  # (5, 23)
        print(f"  actions: {sample['actions'].shape}")  # (50, 7)

        # Verify shapes
        assert sample['video'].shape[0] == 16, "Wrong video length"
        assert sample['proprio'].shape[0] == 5, "Wrong proprio history"
        assert sample['actions'].shape[0] == 50, "Wrong action chunk size"

        print("\n✅ Dataset test passed!")
    else:
        print("No samples found")


if __name__ == "__main__":
    test_libero_dataset()
