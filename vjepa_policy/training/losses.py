"""
Weighted Action Loss Functions

Addresses key failure modes identified in diagnostic analysis:
1. High position error at trajectory start (0.5 vs 0.15 tolerance)
2. High gripper timing error (up to 1.99 at transitions)

Weighted loss applies:
- 3x weight to first 10 timesteps (reduce start error)
- 5x weight to gripper transitions (improve timing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class WeightedActionLoss(nn.Module):
    """
    Weighted L1 loss for action chunks.

    Applies higher weights to:
    1. First N timesteps (trajectory start - where error is highest)
    2. Gripper transitions (where value changes significantly)

    This addresses the behavior cloning generalization gap where:
    - Start of trajectory has 0.5 error (vs 0.15 tolerance)
    - Gripper transitions have up to 1.99 error
    """

    def __init__(
        self,
        start_weight: float = 3.0,
        start_timesteps: int = 10,
        gripper_transition_weight: float = 5.0,
        gripper_dim: int = 6,  # 0-indexed: actions[:, :, 6] is gripper
        transition_threshold: float = 0.5,
        base_weight: float = 1.0,
    ):
        """
        Args:
            start_weight: Weight multiplier for first N timesteps (default 3x)
            start_timesteps: Number of timesteps to apply start weight (default 10)
            gripper_transition_weight: Weight for gripper transitions (default 5x)
            gripper_dim: Index of gripper dimension in actions (default 6)
            transition_threshold: Min change to be considered a transition (default 0.5)
            base_weight: Base weight for all timesteps (default 1.0)
        """
        super().__init__()
        self.start_weight = start_weight
        self.start_timesteps = start_timesteps
        self.gripper_transition_weight = gripper_transition_weight
        self.gripper_dim = gripper_dim
        self.transition_threshold = transition_threshold
        self.base_weight = base_weight

    def compute_temporal_weights(
        self,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-timestep weights based on position and gripper transitions.

        Args:
            target_actions: (B, T, action_dim) target action chunk

        Returns:
            weights: (B, T) temporal weights
        """
        B, T, D = target_actions.shape
        device = target_actions.device

        # Start with base weights
        weights = torch.ones(B, T, device=device) * self.base_weight

        # Apply higher weight to start timesteps
        weights[:, :self.start_timesteps] *= self.start_weight

        # Detect gripper transitions and apply higher weight
        if self.gripper_dim < D:
            gripper = target_actions[:, :, self.gripper_dim]  # (B, T)

            # Compute absolute change from previous timestep
            # Pad first timestep with zeros (no transition at start)
            gripper_diff = torch.zeros_like(gripper)
            gripper_diff[:, 1:] = torch.abs(gripper[:, 1:] - gripper[:, :-1])

            # Mark transitions where change exceeds threshold
            transitions = gripper_diff > self.transition_threshold  # (B, T)

            # Apply transition weight (multiplicative with existing weights)
            weights = weights + transitions.float() * (self.gripper_transition_weight - 1.0)

        return weights

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted L1 loss.

        Args:
            pred_actions: (B, T, action_dim) predicted actions
            target_actions: (B, T, action_dim) ground truth actions

        Returns:
            Dictionary with:
                - 'loss': scalar weighted loss
                - 'loss_unweighted': scalar unweighted L1 loss (for comparison)
                - 'loss_per_dim': (action_dim,) loss per action dimension
                - 'weights_mean': mean temporal weight applied
        """
        B, T, D = pred_actions.shape

        # Compute element-wise L1 loss
        l1_loss = torch.abs(pred_actions - target_actions)  # (B, T, D)

        # Compute temporal weights
        weights = self.compute_temporal_weights(target_actions)  # (B, T)

        # Expand weights to match action dimensions
        weights_expanded = weights.unsqueeze(-1).expand_as(l1_loss)  # (B, T, D)

        # Weighted loss
        weighted_loss = (l1_loss * weights_expanded).sum() / (weights_expanded.sum() + 1e-8)

        # Unweighted loss for comparison
        unweighted_loss = l1_loss.mean()

        # Per-dimension loss (unweighted)
        loss_per_dim = l1_loss.mean(dim=(0, 1))  # (D,)

        return {
            'loss': weighted_loss,
            'loss_unweighted': unweighted_loss,
            'loss_per_dim': loss_per_dim,
            'weights_mean': weights.mean(),
        }


class L1ActionLoss(nn.Module):
    """Standard L1 loss for actions (baseline)."""

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute standard L1 loss.

        Args:
            pred_actions: (B, T, action_dim) predicted actions
            target_actions: (B, T, action_dim) ground truth actions

        Returns:
            Dictionary with 'loss' and 'loss_per_dim'
        """
        loss = F.l1_loss(pred_actions, target_actions)
        loss_per_dim = F.l1_loss(pred_actions, target_actions, reduction='none').mean(dim=(0, 1))

        return {
            'loss': loss,
            'loss_per_dim': loss_per_dim,
        }


class GripperBalancedLoss(nn.Module):
    """
    Loss function with gripper-specific weighting and class balancing.

    Combines:
    1. Temporal weighting (start + transitions)
    2. Class balancing for OPEN vs CLOSE (addresses class imbalance)
    3. Extra weight for gripper dimension
    4. Support for per-sample weights from GripperAugmentationPipeline

    This loss is designed to fix the gripper timing issue where the model
    closes the gripper ~20-30 steps too early.
    """

    def __init__(
        self,
        gripper_loss_weight: float = 2.0,
        balance_gripper: bool = True,
        start_weight: float = 3.0,
        start_timesteps: int = 10,
        gripper_transition_weight: float = 5.0,
        gripper_dim: int = 6,
        transition_threshold: float = 0.5,
    ):
        """
        Args:
            gripper_loss_weight: Extra weight for gripper dimension (default 2×)
            balance_gripper: Balance OPEN vs CLOSE samples (default True)
            start_weight: Weight for first N timesteps (default 3×)
            start_timesteps: Number of timesteps for start weight (default 10)
            gripper_transition_weight: Weight for transitions (default 5×)
            gripper_dim: Index of gripper in actions (default 6)
            transition_threshold: Min change for transition detection
        """
        super().__init__()
        self.gripper_loss_weight = gripper_loss_weight
        self.balance_gripper = balance_gripper
        self.start_weight = start_weight
        self.start_timesteps = start_timesteps
        self.gripper_transition_weight = gripper_transition_weight
        self.gripper_dim = gripper_dim
        self.transition_threshold = transition_threshold

    def compute_temporal_weights(
        self,
        target_actions: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-timestep weights.

        Args:
            target_actions: (B, T, D) target actions
            sample_weights: (B, T) optional weights from augmentation

        Returns:
            weights: (B, T) combined temporal weights
        """
        B, T, D = target_actions.shape
        device = target_actions.device

        # Base weights with start weighting
        weights = torch.ones(B, T, device=device)
        weights[:, :self.start_timesteps] = self.start_weight

        # Detect gripper transitions
        if self.gripper_dim < D:
            gripper = target_actions[:, :, self.gripper_dim]
            gripper_diff = torch.zeros_like(gripper)
            gripper_diff[:, 1:] = torch.abs(gripper[:, 1:] - gripper[:, :-1])
            transitions = gripper_diff > self.transition_threshold
            weights = weights + transitions.float() * (self.gripper_transition_weight - 1.0)

        # Combine with sample-level weights from augmentation
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
            # sample_weights is (B, T), multiply element-wise
            weights = weights * sample_weights

        return weights

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss with gripper balancing.

        Args:
            pred_actions: (B, T, D) predicted actions
            target_actions: (B, T, D) ground truth actions
            sample_weights: (B, T) optional weights from GripperAugmentationPipeline

        Returns:
            Dictionary with loss values
        """
        B, T, D = pred_actions.shape
        device = pred_actions.device

        # Compute temporal weights
        weights = self.compute_temporal_weights(target_actions, sample_weights)

        # Split position/rotation and gripper
        pred_pos_rot = pred_actions[:, :, :self.gripper_dim]
        pred_gripper = pred_actions[:, :, self.gripper_dim]
        gt_pos_rot = target_actions[:, :, :self.gripper_dim]
        gt_gripper = target_actions[:, :, self.gripper_dim]

        # Position/rotation loss (L1)
        pos_rot_loss = torch.abs(pred_pos_rot - gt_pos_rot).mean(dim=-1)  # (B, T)

        # Gripper loss with class balancing
        gripper_loss = torch.abs(pred_gripper - gt_gripper)  # (B, T)

        if self.balance_gripper:
            # Count OPEN (< 0) and CLOSE (> 0) samples
            n_open = (gt_gripper < 0).float().sum() + 1e-6
            n_close = (gt_gripper > 0).float().sum() + 1e-6
            total = n_open + n_close

            # Inverse frequency weights
            weight_open = total / (2 * n_open)
            weight_close = total / (2 * n_close)

            # Apply class weights
            gripper_class_weights = torch.where(gt_gripper < 0, weight_open, weight_close)
            gripper_loss = gripper_loss * gripper_class_weights

        # Combined loss
        combined_loss = pos_rot_loss + self.gripper_loss_weight * gripper_loss

        # Apply temporal weights
        weights_expanded = weights
        weighted_loss = (combined_loss * weights_expanded).sum() / (weights_expanded.sum() + 1e-8)

        # Unweighted loss for logging
        unweighted_loss = combined_loss.mean()

        # Per-dimension loss
        l1_loss = torch.abs(pred_actions - target_actions)
        loss_per_dim = l1_loss.mean(dim=(0, 1))

        return {
            'loss': weighted_loss,
            'loss_unweighted': unweighted_loss,
            'loss_per_dim': loss_per_dim,
            'pos_rot_loss': pos_rot_loss.mean(),
            'gripper_loss': gripper_loss.mean(),
            'weights_mean': weights.mean(),
        }


def get_loss_fn(
    loss_type: str = "weighted",
    **kwargs,
) -> nn.Module:
    """
    Factory function to get loss function.

    Args:
        loss_type: "weighted", "gripper_balanced", or "l1"
        **kwargs: Arguments passed to loss constructor

    Returns:
        Loss module
    """
    if loss_type == "weighted":
        return WeightedActionLoss(**kwargs)
    elif loss_type == "gripper_balanced":
        return GripperBalancedLoss(**kwargs)
    elif loss_type == "l1":
        return L1ActionLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'weighted', 'gripper_balanced', or 'l1'")
