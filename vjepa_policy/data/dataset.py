"""
Unified Dataset for V-JEPA 2 Policy Training.

Single dataset class that handles all augmentation options via configuration:
- L2 normalization of embeddings
- Gaussian noise injection
- Gripper transition oversampling
- Gripper transition temporal jitter
- Per-timestep loss weights
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import Optional, Dict, List, Tuple


class PolicyDataset(Dataset):
    """
    Unified dataset class for V-JEPA 2 policy training with spatial tokens.

    Features (all configurable):
    - L2 normalization of embeddings
    - Gaussian noise injection
    - Gripper transition oversampling
    - Gripper transition temporal jitter
    - Per-timestep loss weights

    Args:
        embeddings_dir: Path to precomputed spatial embeddings
        chunk_size: Action chunk size
        normalize: L2 normalize embeddings
        noise_std: Gaussian noise std (0 = disabled)
        gripper_oversample: Oversample transitions (1 = disabled)
        gripper_jitter: Temporal jitter range (0 = disabled)
        transition_weight: Loss weight for transitions
        transition_window: Window around transitions for weighting
        start_weight: Loss weight for initial timesteps
        start_steps: Number of initial steps to upweight
        is_training: Enable augmentations
    """

    def __init__(
        self,
        embeddings_dir: str,
        chunk_size: int = 50,
        # Embedding augmentation
        normalize: bool = True,
        noise_std: float = 0.05,
        # Gripper augmentation
        gripper_oversample: int = 1,
        gripper_jitter: int = 0,
        # Loss weighting
        transition_weight: float = 5.0,
        transition_window: int = 10,
        start_weight: float = 3.0,
        start_steps: int = 10,
        # Mode
        is_training: bool = True,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.noise_std = noise_std
        self.gripper_oversample = gripper_oversample
        self.gripper_jitter = gripper_jitter
        self.transition_weight = transition_weight
        self.transition_window = transition_window
        self.start_weight = start_weight
        self.start_steps = start_steps
        self.is_training = is_training

        # Load all samples into memory
        self.samples = self._load_samples()

        # Cache transitions and weights for each sample (computed once, not per iteration)
        self._cache_sample_metadata()

        # Build index with optional oversampling
        self.indices = self._build_indices()

        print(f"Loaded {len(self.samples)} base samples from {self.embeddings_dir}")
        if gripper_oversample > 1 and is_training:
            n_trans = sum(1 for i in self.indices if self._sample_has_transition[i])
            print(f"After {gripper_oversample}x oversampling: {len(self.indices)} samples")
            print(f"Transition samples: {n_trans} ({100*n_trans/len(self.indices):.1f}%)")

    def _load_samples(self) -> List[Dict]:
        """Load all precomputed embedding files."""
        samples = []

        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {self.embeddings_dir}")

        files = sorted(self.embeddings_dir.glob("*.pt"))
        if len(files) == 0:
            raise FileNotFoundError(f"No .pt files found in {self.embeddings_dir}")

        for path in files:
            data = torch.load(path, map_location='cpu')

            # Support both naming conventions
            video_tokens = data.get('video_tokens', data.get('video_emb', data.get('current_emb')))
            goal_tokens = data.get('goal_tokens', data.get('goal_emb'))

            sample = {
                'video_tokens': video_tokens,
                'goal_tokens': goal_tokens,
                'proprio': data['proprio'],
                'actions': data['actions'],
            }
            samples.append(sample)

        return samples

    def _cache_sample_metadata(self):
        """Cache transitions and weights for all samples at init time."""
        self._sample_transitions = []  # List of transition indices per sample
        self._sample_has_transition = []  # Boolean per sample
        self._sample_weights = []  # Cached weights per sample

        for sample in self.samples:
            actions = sample['actions'][:self.chunk_size]

            # Pad actions if needed (same logic as __getitem__)
            if len(actions) < self.chunk_size:
                pad_len = self.chunk_size - len(actions)
                pad = actions[-1:].repeat(pad_len, 1)
                actions = torch.cat([actions, pad], dim=0)

            # Find transitions
            transitions = self._find_transitions(actions)
            self._sample_transitions.append(transitions)
            self._sample_has_transition.append(len(transitions) > 0)

            # Compute weights (using vectorized approach)
            weights = self._compute_weights_vectorized(actions, transitions)
            self._sample_weights.append(weights)

    def _compute_weights_vectorized(self, actions: torch.Tensor, transitions: List[int]) -> torch.Tensor:
        """Compute weights using vectorized operations (faster than per-element)."""
        T = len(actions)
        weights = torch.ones(T)

        # Start weighting
        weights[:self.start_steps] = self.start_weight

        # Transition weighting (vectorized)
        if transitions:
            for trans_t in transitions:
                start = max(0, trans_t - self.transition_window)
                end = min(T, trans_t + self.transition_window + 1)
                weights[start:end] = torch.clamp(
                    weights[start:end],
                    min=self.transition_weight
                )

        return weights

    def _has_transition(self, actions: torch.Tensor) -> bool:
        """Check if action sequence contains gripper transition."""
        gripper = actions[:, 6]
        for t in range(1, len(gripper)):
            if abs(gripper[t].item() - gripper[t-1].item()) > 0.5:
                return True
        return False

    def _find_transitions(self, actions: torch.Tensor) -> List[int]:
        """Find timesteps where gripper transitions."""
        gripper = actions[:, 6]
        transitions = []
        for t in range(1, len(gripper)):
            if abs(gripper[t].item() - gripper[t-1].item()) > 0.5:
                transitions.append(t)
        return transitions

    def _build_indices(self) -> List[int]:
        """Build sample indices with optional gripper oversampling."""
        if self.gripper_oversample <= 1 or not self.is_training:
            return list(range(len(self.samples)))

        indices = []
        for i in range(len(self.samples)):
            if self._sample_has_transition[i]:
                # Oversample transitions
                indices.extend([i] * self.gripper_oversample)
            else:
                indices.append(i)

        return indices

    def _apply_gripper_jitter(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply temporal jitter to gripper transitions."""
        if self.gripper_jitter <= 0 or not self.is_training:
            return actions

        actions = actions.clone()
        T = len(actions)
        transitions = self._find_transitions(actions)

        for trans_t in transitions:
            jitter = random.randint(-self.gripper_jitter, self.gripper_jitter)
            if jitter == 0:
                continue

            new_t = max(1, min(T - 1, trans_t + jitter))
            gripper_before = actions[trans_t - 1, 6].item()
            gripper_after = actions[trans_t, 6].item()

            if jitter > 0:
                # Delay transition: keep old value longer
                actions[trans_t:new_t, 6] = gripper_before
            else:
                # Earlier transition: new value starts sooner
                actions[new_t:trans_t, 6] = gripper_after

        return actions

    def _compute_weights(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep loss weights."""
        T = len(actions)
        weights = torch.ones(T)

        # Start weighting - higher weight on initial timesteps
        weights[:self.start_steps] = self.start_weight

        # Transition weighting - higher weight around gripper transitions
        transitions = self._find_transitions(actions)
        for trans_t in transitions:
            start = max(0, trans_t - self.transition_window)
            end = min(T, trans_t + self.transition_window + 1)
            weights[start:end] = torch.maximum(
                weights[start:end],
                torch.tensor(self.transition_weight)
            )

        return weights

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]

        # Get data - spatial tokens (64, 1408)
        video_tokens = sample['video_tokens']
        goal_tokens = sample['goal_tokens']
        proprio = sample['proprio']
        actions = sample['actions'][:self.chunk_size]

        # Normalize proprio shape - some datasets have [5, 15], others have [75]
        # Always use [5, 15] format for the model
        if proprio.dim() == 1:
            # Flattened format [75] -> reshape to [5, 15]
            proprio = proprio.view(5, 15)
        # Already [5, 15] - no change needed

        # Pad actions if needed
        if len(actions) < self.chunk_size:
            pad_len = self.chunk_size - len(actions)
            pad = actions[-1:].repeat(pad_len, 1)
            actions = torch.cat([actions, pad], dim=0)

        # Use cached weights (clone only if training with jitter to avoid modifying cache)
        if self.gripper_jitter > 0 and self.is_training:
            actions = actions.clone()
            actions = self._apply_gripper_jitter(actions)
            # Recompute weights if jitter was applied (transitions may have moved)
            transitions = self._find_transitions(actions)
            weights = self._compute_weights_vectorized(actions, transitions)
        else:
            # Use cached weights (much faster)
            weights = self._sample_weights[sample_idx]

        # Apply normalization
        if self.normalize:
            video_tokens = F.normalize(video_tokens, dim=-1)
            goal_tokens = F.normalize(goal_tokens, dim=-1)

        # Apply noise (training only) - clone here if noise is applied
        if self.noise_std > 0 and self.is_training:
            video_tokens = video_tokens + torch.randn_like(video_tokens) * self.noise_std
            goal_tokens = goal_tokens + torch.randn_like(goal_tokens) * self.noise_std

        return {
            'video_tokens': video_tokens,
            'goal_tokens': goal_tokens,
            'proprio': proprio,
            'actions': actions,
            'weights': weights,
        }

    def train(self):
        """Enable training mode (augmentations on)."""
        self.is_training = True

    def eval(self):
        """Enable eval mode (augmentations off)."""
        self.is_training = False


class CombinedDataset(Dataset):
    """Dataset that combines multiple embedding directories (for multi-suite training)."""

    def __init__(self, datasets: List[PolicyDataset]):
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._total = sum(self._lengths)
        # Build cumulative offsets for indexing
        self._offsets = [0]
        for L in self._lengths[:-1]:
            self._offsets.append(self._offsets[-1] + L)

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, (offset, length) in enumerate(zip(self._offsets, self._lengths)):
            if idx < offset + length:
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range")

    def train(self):
        for d in self.datasets:
            d.train()

    def eval(self):
        for d in self.datasets:
            d.eval()


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders from config.

    Args:
        config: Full configuration dict with 'data', 'model', 'loss', 'training' sections

    Returns:
        train_loader, val_loader

    Supports multi-suite training via 'suites' config key (list of suite names).
    """
    data_cfg = config['data']
    model_cfg = config['model']
    loss_cfg = config.get('loss', {})
    train_cfg = config['training']

    # Determine embeddings directory
    # Support multiple suites via 'suites' list, or single suite via 'suite'
    suites = data_cfg.get('suites', [data_cfg.get('suite', 'libero_spatial')])
    if isinstance(suites, str):
        suites = [suites]
    base_dir = Path(data_cfg['embeddings_dir'])

    # Collect datasets from all suites
    train_datasets = []
    val_datasets = []

    for suite in suites:
        # Try suite-specific subdirectory first
        suite_dir = base_dir / f"{suite}_spatial"
        if not suite_dir.exists():
            suite_dir = base_dir / suite
        if not suite_dir.exists():
            suite_dir = base_dir

        # Create train dataset for this suite
        train_dir = suite_dir / 'train'
        if not train_dir.exists():
            train_dir = suite_dir  # Fall back to non-split directory

        if train_dir.exists() and len(list(train_dir.glob("*.pt"))) > 0:
            train_datasets.append(PolicyDataset(
                embeddings_dir=str(train_dir),
                chunk_size=model_cfg.get('chunk_size', 50),
                normalize=data_cfg.get('normalize', True),
                noise_std=data_cfg.get('noise_std', 0.05),
                gripper_oversample=data_cfg.get('gripper_oversample', 1),
                gripper_jitter=data_cfg.get('gripper_jitter', 0),
                transition_weight=loss_cfg.get('transition_weight', 5.0),
                transition_window=loss_cfg.get('transition_window', 10),
                start_weight=loss_cfg.get('start_weight', 3.0),
                start_steps=loss_cfg.get('start_steps', 10),
                is_training=True,
            ))

        # Create val dataset for this suite
        val_dir = suite_dir / 'val'
        if val_dir.exists() and len(list(val_dir.glob("*.pt"))) > 0:
            val_datasets.append(PolicyDataset(
                embeddings_dir=str(val_dir),
                chunk_size=model_cfg.get('chunk_size', 50),
                normalize=data_cfg.get('normalize', True),
                noise_std=0.0,  # No noise for validation
                gripper_oversample=1,  # No oversampling for validation
                gripper_jitter=0,  # No jitter for validation
                is_training=False,
            ))

    if len(train_datasets) == 0:
        raise FileNotFoundError(f"No training data found for suites {suites} in {base_dir}")

    # Combine datasets if multiple suites
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = CombinedDataset(train_datasets)
        print(f"Combined {len(train_datasets)} suites for training")

    if len(val_datasets) == 0:
        print("Warning: No val split found, using train data for validation")
        val_dataset = train_datasets[0] if len(train_datasets) == 1 else CombinedDataset([d for d in train_datasets])
    elif len(val_datasets) == 1:
        val_dataset = val_datasets[0]
    else:
        val_dataset = CombinedDataset(val_datasets)
        print(f"Combined {len(val_datasets)} suites for validation")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
    )

    return train_loader, val_loader


def test_dataset():
    """Test dataset loading."""
    import tempfile
    import os

    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake samples
        for i in range(10):
            sample = {
                'video_tokens': torch.randn(64, 1408),
                'goal_tokens': torch.randn(64, 1408),
                'proprio': torch.randn(75),  # 15 * 5
                'actions': torch.randn(50, 7),
            }
            # Add a transition to some samples
            if i % 3 == 0:
                sample['actions'][25:, 6] = 1.0  # gripper closes
                sample['actions'][:25, 6] = -1.0  # gripper open
            torch.save(sample, os.path.join(tmpdir, f"sample_{i:06d}.pt"))

        # Test dataset
        dataset = PolicyDataset(
            embeddings_dir=tmpdir,
            normalize=True,
            noise_std=0.05,
            gripper_oversample=5,
            gripper_jitter=5,
            is_training=True,
        )

        print(f"Dataset length: {len(dataset)}")

        # Get a sample
        sample = dataset[0]
        print(f"video_tokens: {sample['video_tokens'].shape}")
        print(f"goal_tokens: {sample['goal_tokens'].shape}")
        print(f"proprio: {sample['proprio'].shape}")
        print(f"actions: {sample['actions'].shape}")
        print(f"weights: {sample['weights'].shape}")

        # Test dataloader
        loader = DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")

        print("\nDataset test passed!")


if __name__ == "__main__":
    test_dataset()
