"""
LIBERO Dataset Loader

Loads demonstrations from LIBERO benchmark for training.
Supports pre-computed embeddings for faster training.
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class LIBERODataset(Dataset):
    """
    LIBERO demonstration dataset.
    
    Each sample contains:
    - video: 16 frames of observation
    - goal: final frame of demonstration
    - proprio: proprioception history
    - actions: action chunk (50 actions)
    """
    
    def __init__(
        self,
        data_dir: str,
        suite: str = "libero_object",
        split: str = "train",
        video_len: int = 16,
        proprio_history: int = 5,
        chunk_size: int = 50,
        image_size: int = 256,
        precomputed_emb_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Path to LIBERO data
            suite: One of libero_object, libero_spatial, libero_goal, libero_long
            split: train or val
            video_len: Number of frames for video input
            proprio_history: Number of proprio frames
            chunk_size: Number of actions to predict
            image_size: Image resolution
            precomputed_emb_dir: Path to pre-computed embeddings (optional)
        """
        self.data_dir = Path(data_dir)
        self.suite = suite
        self.split = split
        self.video_len = video_len
        self.proprio_history = proprio_history
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.precomputed_emb_dir = Path(precomputed_emb_dir) if precomputed_emb_dir else None
        
        # Load demo metadata
        self.demos = self._load_demos()
        
        # Create index of valid starting points
        self.samples = self._create_sample_index()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.demos)} demos")
    
    def _load_demos(self) -> List[Dict]:
        """Load all demonstrations for the suite"""
        demos = []
        
        suite_dir = self.data_dir / self.suite
        demo_files = sorted(suite_dir.glob("*.hdf5"))
        
        for demo_file in demo_files:
            with h5py.File(demo_file, 'r') as f:
                # Get demo info
                demo_data = {
                    'path': str(demo_file),
                    'length': f['actions'].shape[0],
                    'task_name': f.attrs.get('task_name', 'unknown'),
                }
                demos.append(demo_data)
        
        return demos
    
    def _create_sample_index(self) -> List[Tuple[int, int]]:
        """Create index of (demo_idx, start_frame) pairs"""
        samples = []
        
        min_length = self.video_len + self.chunk_size
        
        for demo_idx, demo in enumerate(self.demos):
            demo_len = demo['length']
            
            if demo_len < min_length:
                continue
            
            # Can start from frame 0 to (demo_len - min_length)
            max_start = demo_len - min_length
            
            # Sample starting points (every 5 frames to reduce overlap)
            for start in range(0, max_start, 5):
                samples.append((demo_idx, start))
        
        return samples
    
    def _load_demo_data(self, demo_path: str) -> Dict:
        """Load full demo data from HDF5 file"""
        with h5py.File(demo_path, 'r') as f:
            data = {
                'images': f['obs/agentview_image'][:],  # (T, H, W, 3)
                'actions': f['actions'][:],              # (T, 7)
                'proprio': self._extract_proprio(f),     # (T, 23)
            }
        return data
    
    def _extract_proprio(self, f: h5py.File) -> np.ndarray:
        """Extract proprioception from HDF5 file"""
        # Concatenate all proprio components
        proprio_keys = [
            'obs/robot0_eef_pos',       # (T, 3)
            'obs/robot0_eef_quat',      # (T, 4)
            'obs/robot0_gripper_qpos',  # (T, 2)
            'obs/robot0_joint_pos',     # (T, 7)
            'obs/robot0_joint_vel',     # (T, 7)
        ]
        
        proprio_parts = []
        for key in proprio_keys:
            if key in f:
                proprio_parts.append(f[key][:])
        
        return np.concatenate(proprio_parts, axis=-1)  # (T, 23)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo_idx, start_frame = self.samples[idx]
        demo = self.demos[demo_idx]
        
        # Load demo data
        demo_data = self._load_demo_data(demo['path'])
        
        # Extract video frames
        video_end = start_frame + self.video_len
        video = demo_data['images'][start_frame:video_end]  # (video_len, H, W, 3)
        
        # Extract goal (last frame of demo)
        goal = demo_data['images'][-1]  # (H, W, 3)
        
        # Extract proprio history
        proprio_start = max(0, start_frame - self.proprio_history + 1)
        proprio = demo_data['proprio'][proprio_start:start_frame + 1]
        
        # Pad proprio if needed
        if len(proprio) < self.proprio_history:
            pad = np.repeat(proprio[:1], self.proprio_history - len(proprio), axis=0)
            proprio = np.concatenate([pad, proprio], axis=0)
        
        # Extract action chunk
        action_start = start_frame + self.video_len - 1  # Start from last video frame
        action_end = action_start + self.chunk_size
        actions = demo_data['actions'][action_start:action_end]
        
        # Pad actions if needed
        if len(actions) < self.chunk_size:
            pad = np.repeat(actions[-1:], self.chunk_size - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        
        # Convert to tensors
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        goal = torch.from_numpy(goal).permute(2, 0, 1).float() / 255.0       # (C, H, W)
        proprio = torch.from_numpy(proprio).float()                           # (history, 23)
        actions = torch.from_numpy(actions).float()                           # (chunk, 7)
        
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
    """
    
    def __init__(
        self,
        embedding_dir: str,
        proprio_history: int = 5,
        chunk_size: int = 50,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.proprio_history = proprio_history
        self.chunk_size = chunk_size
        
        # Load all embedding files
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} pre-computed samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata"""
        samples = []
        
        for emb_file in sorted(self.embedding_dir.glob("*.pt")):
            data = torch.load(emb_file)
            samples.append({
                'path': str(emb_file),
                'length': data['current_emb'].shape[0],
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_path = self.samples[idx]['path']
        data = torch.load(sample_path)
        
        return {
            'current_emb': data['current_emb'],
            'goal_emb': data['goal_emb'],
            'proprio': data['proprio'],
            'actions': data['actions'],
        }


def create_dataloader(
    data_dir: str,
    suite: str = "libero_object",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create LIBERO dataloader"""
    
    dataset = LIBERODataset(data_dir=data_dir, suite=suite, **kwargs)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader