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


def get_loss_fn(
    loss_type: str = "weighted",
    **kwargs,
) -> nn.Module:
    """
    Factory function to get loss function.

    Args:
        loss_type: "weighted" or "l1"
        **kwargs: Arguments passed to loss constructor

    Returns:
        Loss module
    """
    if loss_type == "weighted":
        return WeightedActionLoss(**kwargs)
    elif loss_type == "l1":
        return L1ActionLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'weighted' or 'l1'")
