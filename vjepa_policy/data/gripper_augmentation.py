"""
Gripper-focused data augmentation for improving gripper timing predictions.

Key components:
1. GripperAugmentationPipeline: Applies jitter, noise, and computes weights
2. GripperFocusedSampler: Oversamples transitions during training
3. compute_gripper_weighted_loss: Loss function with transition focus

Root cause: Gripper closes ~20-30 steps too early because:
- Gripper transitions are RARE in training data (~5% of timesteps)
- Model sees mostly stable gripper states (open or closed)
- When averaging over demos with different transition timings, model learns wrong average

Solution: Oversample and augment gripper transition moments.
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class GripperAugmentationPipeline:
    """
    Pipeline for gripper-focused data augmentation.

    Augmentations:
    1. Temporal jitter: Shift gripper transitions by ±jitter_range steps
    2. Noise injection: Add small noise to gripper values
    3. Transition weighting: Compute per-timestep weights for loss
    """

    def __init__(
        self,
        jitter_range: int = 5,
        jitter_prob: float = 0.5,
        noise_std: float = 0.05,
        noise_prob: float = 0.3,
        transition_weight: float = 5.0,
        transition_window: int = 10,
    ):
        """
        Args:
            jitter_range: Max steps to shift transitions (±)
            jitter_prob: Probability of applying jitter
            noise_std: Std of Gaussian noise for gripper values
            noise_prob: Probability of applying noise
            transition_weight: Weight multiplier for transition timesteps
            transition_window: Window around transition to upweight (±steps)
        """
        self.jitter_range = jitter_range
        self.jitter_prob = jitter_prob
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.transition_weight = transition_weight
        self.transition_window = transition_window

    def find_transitions(self, actions: torch.Tensor) -> List[int]:
        """
        Find timesteps where gripper changes state.

        Args:
            actions: (T, 7) action sequence, dim 6 is gripper

        Returns:
            List of transition timesteps
        """
        gripper = actions[:, 6]
        transitions = []

        for t in range(1, len(gripper)):
            # Detect significant change (open ↔ close)
            if abs(gripper[t].item() - gripper[t-1].item()) > 0.5:
                transitions.append(t)

        return transitions

    def apply_jitter(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal jitter to gripper transitions.

        Randomly shifts when the gripper transition happens by ±jitter_range steps.
        This teaches the model that exact transition timing can vary.

        Args:
            actions: (T, 7) action sequence

        Returns:
            Augmented actions with shifted transitions
        """
        if random.random() > self.jitter_prob:
            return actions

        actions = actions.clone()
        T = len(actions)
        transitions = self.find_transitions(actions)

        if len(transitions) == 0:
            return actions

        for trans_t in transitions:
            # Random jitter
            jitter = random.randint(-self.jitter_range, self.jitter_range)
            new_trans_t = max(1, min(T - 1, trans_t + jitter))

            if jitter == 0:
                continue

            # Get gripper values before and after transition
            gripper_before = actions[trans_t - 1, 6].item()
            gripper_after = actions[trans_t, 6].item()

            if jitter > 0:
                # Delay transition: Extend previous state forward
                actions[trans_t:new_trans_t, 6] = gripper_before
            else:
                # Advance transition: Extend new state backward
                actions[new_trans_t:trans_t, 6] = gripper_after

        return actions

    def apply_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Add small Gaussian noise to gripper values.

        Helps model be robust to slight variations in gripper state.

        Args:
            actions: (T, 7) action sequence

        Returns:
            Actions with noisy gripper values
        """
        if random.random() > self.noise_prob:
            return actions

        actions = actions.clone()
        noise = torch.randn(actions.shape[0]) * self.noise_std
        actions[:, 6] = torch.clamp(actions[:, 6] + noise, -1.0, 1.0)

        return actions

    def compute_weights(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute per-timestep loss weights.

        Higher weight around gripper transitions to focus learning.

        Args:
            actions: (T, 7) action sequence

        Returns:
            weights: (T,) per-timestep weights
        """
        T = len(actions)
        weights = torch.ones(T)

        transitions = self.find_transitions(actions)

        for trans_t in transitions:
            # Apply higher weight in window around transition
            start = max(0, trans_t - self.transition_window)
            end = min(T, trans_t + self.transition_window + 1)
            weights[start:end] = self.transition_weight

        return weights

    def __call__(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all augmentations and compute weights.

        Args:
            actions: (T, 7) action sequence

        Returns:
            augmented_actions: (T, 7)
            weights: (T,)
        """
        # Apply augmentations
        actions = self.apply_jitter(actions)
        actions = self.apply_noise(actions)

        # Compute weights (on augmented actions)
        weights = self.compute_weights(actions)

        return actions, weights


def compute_gripper_weighted_loss(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    gripper_loss_weight: float = 2.0,
    balance_gripper: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss with gripper-specific weighting.

    Args:
        pred_actions: (B, T, 7) predicted actions
        gt_actions: (B, T, 7) ground truth actions
        weights: (B, T) per-timestep weights (from augmentation pipeline)
        gripper_loss_weight: Extra weight for gripper dimension
        balance_gripper: Whether to balance OPEN vs CLOSE samples

    Returns:
        Dictionary with loss values
    """
    B, T, D = pred_actions.shape
    device = pred_actions.device

    # Split position/rotation and gripper
    pred_pos_rot = pred_actions[:, :, :6]  # (B, T, 6)
    pred_gripper = pred_actions[:, :, 6]    # (B, T)
    gt_pos_rot = gt_actions[:, :, :6]       # (B, T, 6)
    gt_gripper = gt_actions[:, :, 6]        # (B, T)

    # Position/rotation loss (L1)
    pos_rot_loss = F.l1_loss(pred_pos_rot, gt_pos_rot, reduction='none')  # (B, T, 6)
    pos_rot_loss = pos_rot_loss.mean(dim=-1)  # (B, T)

    # Gripper loss (L1)
    gripper_loss = F.l1_loss(pred_gripper, gt_gripper, reduction='none')  # (B, T)

    # Balance gripper classes if requested
    if balance_gripper:
        # Count OPEN (< 0) and CLOSE (> 0) samples
        n_open = (gt_gripper < 0).float().sum() + 1e-6
        n_close = (gt_gripper > 0).float().sum() + 1e-6
        total = n_open + n_close

        # Inverse frequency weights
        weight_open = total / (2 * n_open)
        weight_close = total / (2 * n_close)

        # Apply class weights to gripper loss
        gripper_weights = torch.where(gt_gripper < 0, weight_open, weight_close)
        gripper_loss = gripper_loss * gripper_weights

    # Combine losses
    combined_loss = pos_rot_loss + gripper_loss_weight * gripper_loss  # (B, T)

    # Apply per-timestep weights if provided
    if weights is not None:
        weights = weights.to(device)
        combined_loss = combined_loss * weights
        loss = combined_loss.sum() / (weights.sum() + 1e-8)
    else:
        loss = combined_loss.mean()

    return {
        'loss': loss,
        'pos_rot_loss': pos_rot_loss.mean(),
        'gripper_loss': gripper_loss.mean(),
    }


class GripperFocusedSampler:
    """
    Sampler that oversamples training windows containing gripper transitions.

    Use with your dataset to create an oversampled index list.
    """

    def __init__(
        self,
        actions_list: List[torch.Tensor],
        chunk_size: int = 50,
        oversample_factor: int = 5,
    ):
        """
        Args:
            actions_list: List of (T, 7) action tensors, one per demo
            chunk_size: Size of action chunks for training
            oversample_factor: How many times to oversample transitions
        """
        self.actions_list = actions_list
        self.chunk_size = chunk_size
        self.oversample_factor = oversample_factor

        self.samples = self._build_sample_list()

    def _find_transitions(self, actions: torch.Tensor) -> List[int]:
        """Find gripper transition timesteps."""
        gripper = actions[:, 6]
        transitions = []
        for t in range(1, len(gripper)):
            if abs(gripper[t].item() - gripper[t-1].item()) > 0.5:
                transitions.append(t)
        return transitions

    def _build_sample_list(self) -> List[Tuple[int, int, bool]]:
        """
        Build list of (demo_idx, start_frame, is_transition) samples.

        Oversamples windows that contain gripper transitions.
        """
        samples = []

        for demo_idx, actions in enumerate(self.actions_list):
            T = len(actions)
            max_start = T - self.chunk_size

            if max_start <= 0:
                continue

            # Find transition timesteps
            transitions = self._find_transitions(actions)

            # Create set of "transition windows" (start frames where chunk contains transition)
            transition_starts = set()
            for trans_t in transitions:
                # Chunk [start, start + chunk_size) contains trans_t if:
                # start <= trans_t < start + chunk_size
                # So: trans_t - chunk_size < start <= trans_t
                min_start = max(0, trans_t - self.chunk_size + 1)
                max_start_for_trans = min(trans_t, max_start)
                for s in range(min_start, max_start_for_trans + 1):
                    transition_starts.add(s)

            # Add all valid start frames
            for start in range(max_start + 1):
                if start in transition_starts:
                    # Oversample transitions
                    for _ in range(self.oversample_factor):
                        samples.append((demo_idx, start, True))
                else:
                    # Normal sampling
                    samples.append((demo_idx, start, False))

        return samples

    def __len__(self):
        return len(self.samples)

    def get_sample_info(self, idx: int) -> Tuple[int, int, bool]:
        """Get (demo_idx, start_frame, is_transition) for sample idx."""
        return self.samples[idx]

    def get_stats(self) -> dict:
        """Get statistics about the sampling."""
        n_transition = sum(1 for _, _, is_trans in self.samples if is_trans)
        n_normal = len(self.samples) - n_transition
        return {
            'total_samples': len(self.samples),
            'transition_samples': n_transition,
            'normal_samples': n_normal,
            'transition_ratio': n_transition / len(self.samples) if len(self.samples) > 0 else 0,
        }


class GripperBalancedLoss(torch.nn.Module):
    """
    Loss function with gripper-specific weighting and class balancing.

    Combines:
    1. Temporal weighting from GripperAugmentationPipeline
    2. Class balancing for OPEN vs CLOSE
    3. Extra weight for gripper dimension
    """

    def __init__(
        self,
        gripper_loss_weight: float = 2.0,
        balance_gripper: bool = True,
        start_weight: float = 3.0,
        start_timesteps: int = 10,
    ):
        super().__init__()
        self.gripper_loss_weight = gripper_loss_weight
        self.balance_gripper = balance_gripper
        self.start_weight = start_weight
        self.start_timesteps = start_timesteps

    def forward(
        self,
        pred_actions: torch.Tensor,
        gt_actions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss.

        Args:
            pred_actions: (B, T, 7) predicted actions
            gt_actions: (B, T, 7) ground truth actions
            weights: (B, T) per-timestep weights from augmentation

        Returns:
            Dictionary with loss values
        """
        B, T, D = pred_actions.shape
        device = pred_actions.device

        # Base temporal weights (start weighting)
        base_weights = torch.ones(B, T, device=device)
        base_weights[:, :self.start_timesteps] = self.start_weight

        # Combine with provided weights (transition weighting)
        if weights is not None:
            weights = weights.to(device)
            combined_weights = base_weights * weights
        else:
            combined_weights = base_weights

        # Split position/rotation and gripper
        pred_pos_rot = pred_actions[:, :, :6]
        pred_gripper = pred_actions[:, :, 6]
        gt_pos_rot = gt_actions[:, :, :6]
        gt_gripper = gt_actions[:, :, 6]

        # Position/rotation loss
        pos_rot_loss = F.l1_loss(pred_pos_rot, gt_pos_rot, reduction='none').mean(dim=-1)

        # Gripper loss with class balancing
        gripper_loss = F.l1_loss(pred_gripper, gt_gripper, reduction='none')

        if self.balance_gripper:
            n_open = (gt_gripper < 0).float().sum() + 1e-6
            n_close = (gt_gripper > 0).float().sum() + 1e-6
            total = n_open + n_close
            weight_open = total / (2 * n_open)
            weight_close = total / (2 * n_close)
            gripper_weights = torch.where(gt_gripper < 0, weight_open, weight_close)
            gripper_loss = gripper_loss * gripper_weights

        # Combined loss with gripper weight
        combined_loss = pos_rot_loss + self.gripper_loss_weight * gripper_loss

        # Apply temporal weights
        weighted_loss = combined_loss * combined_weights
        loss = weighted_loss.sum() / (combined_weights.sum() + 1e-8)

        return {
            'loss': loss,
            'loss_unweighted': combined_loss.mean(),
            'pos_rot_loss': pos_rot_loss.mean(),
            'gripper_loss': gripper_loss.mean(),
            'weights_mean': combined_weights.mean(),
        }


# Utility functions for analysis

def analyze_gripper_transitions(actions_list: List[torch.Tensor]) -> Dict:
    """
    Analyze gripper transitions in a dataset.

    Args:
        actions_list: List of (T, 7) action tensors

    Returns:
        Statistics about gripper transitions
    """
    total_timesteps = 0
    total_transitions = 0
    transition_positions = []

    for actions in actions_list:
        T = len(actions)
        total_timesteps += T

        gripper = actions[:, 6]
        for t in range(1, T):
            if abs(gripper[t].item() - gripper[t-1].item()) > 0.5:
                total_transitions += 1
                transition_positions.append(t / T)  # Normalized position

    return {
        'total_timesteps': total_timesteps,
        'total_transitions': total_transitions,
        'transition_ratio': total_transitions / total_timesteps if total_timesteps > 0 else 0,
        'avg_transition_position': np.mean(transition_positions) if transition_positions else 0,
        'n_demos': len(actions_list),
    }
