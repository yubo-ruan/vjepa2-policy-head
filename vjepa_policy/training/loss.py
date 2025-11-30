"""
Unified Loss Function for V-JEPA 2 Policy Training.

Single loss class with configurable weighting:
- Per-timestep weighting (from dataset)
- Gripper class balancing
- Extra gripper dimension weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ActionLoss(nn.Module):
    """
    Unified action loss with configurable weighting.

    Features:
    - Per-timestep weighting (from dataset weights)
    - Gripper class balancing (OPEN vs CLOSE)
    - Extra gripper dimension weight
    - Temporal weighting (start + transitions)

    Args:
        gripper_loss_weight: Extra weight for gripper dimension
        balance_gripper: Balance OPEN vs CLOSE samples
        gripper_dim: Index of gripper in action vector
        start_weight: Weight multiplier for first N timesteps
        start_steps: Number of timesteps for start weight
        transition_weight: Weight for gripper transitions
        transition_threshold: Min change to detect transition
    """

    def __init__(
        self,
        gripper_loss_weight: float = 2.0,
        balance_gripper: bool = True,
        gripper_dim: int = 6,
        start_weight: float = 3.0,
        start_steps: int = 10,
        transition_weight: float = 5.0,
        transition_threshold: float = 0.5,
    ):
        super().__init__()
        self.gripper_loss_weight = gripper_loss_weight
        self.balance_gripper = balance_gripper
        self.gripper_dim = gripper_dim
        self.start_weight = start_weight
        self.start_steps = start_steps
        self.transition_weight = transition_weight
        self.transition_threshold = transition_threshold

    def compute_temporal_weights(
        self,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-timestep weights based on position and transitions.

        Args:
            target_actions: (B, T, D) target actions

        Returns:
            weights: (B, T) temporal weights
        """
        B, T, D = target_actions.shape
        device = target_actions.device

        # Base weights with start weighting
        weights = torch.ones(B, T, device=device)
        weights[:, :self.start_steps] = self.start_weight

        # Detect gripper transitions
        if self.gripper_dim < D:
            gripper = target_actions[:, :, self.gripper_dim]
            gripper_diff = torch.zeros_like(gripper)
            gripper_diff[:, 1:] = torch.abs(gripper[:, 1:] - gripper[:, :-1])
            transitions = gripper_diff > self.transition_threshold
            weights = weights + transitions.float() * (self.transition_weight - 1.0)

        return weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted action loss.

        Args:
            pred: (B, T, D) predicted actions
            target: (B, T, D) target actions
            weights: (B, T) per-timestep weights (optional, from dataset)

        Returns:
            Dictionary with loss values
        """
        B, T, D = pred.shape
        device = pred.device

        # Split position/rotation and gripper
        pred_pos = pred[:, :, :self.gripper_dim]
        pred_grip = pred[:, :, self.gripper_dim]
        target_pos = target[:, :, :self.gripper_dim]
        target_grip = target[:, :, self.gripper_dim]

        # Position/rotation loss (L1)
        pos_loss = F.l1_loss(pred_pos, target_pos, reduction='none').mean(dim=-1)  # (B, T)

        # Gripper loss (L1)
        grip_loss = F.l1_loss(pred_grip, target_grip, reduction='none')  # (B, T)

        # Balance gripper classes
        if self.balance_gripper:
            n_open = (target_grip < 0).float().sum() + 1e-6
            n_close = (target_grip > 0).float().sum() + 1e-6
            total = n_open + n_close

            weight_open = total / (2 * n_open)
            weight_close = total / (2 * n_close)

            grip_weights = torch.where(target_grip < 0, weight_open, weight_close)
            grip_loss = grip_loss * grip_weights

        # Combine losses
        combined = pos_loss + self.gripper_loss_weight * grip_loss  # (B, T)

        # Get temporal weights (from dataset or computed)
        if weights is not None:
            temporal_weights = weights.to(device)
        else:
            temporal_weights = self.compute_temporal_weights(target)

        # Apply temporal weights
        weighted_loss = (combined * temporal_weights).sum() / (temporal_weights.sum() + 1e-6)

        # Unweighted loss for logging
        unweighted_loss = combined.mean()

        # Per-dimension loss
        l1_loss = torch.abs(pred - target)
        loss_per_dim = l1_loss.mean(dim=(0, 1))

        return {
            'loss': weighted_loss,
            'loss_unweighted': unweighted_loss,
            'loss_per_dim': loss_per_dim,
            'pos_loss': pos_loss.mean(),
            'gripper_loss': grip_loss.mean(),
            'weights_mean': temporal_weights.mean(),
        }

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Allow calling without .forward()"""
        return self.forward(pred, target, weights)


def create_loss(config: dict) -> ActionLoss:
    """Create loss function from config."""
    loss_cfg = config.get('loss', {})
    return ActionLoss(
        gripper_loss_weight=loss_cfg.get('gripper_loss_weight', 2.0),
        balance_gripper=loss_cfg.get('balance_gripper', True),
        gripper_dim=loss_cfg.get('gripper_dim', 6),
        start_weight=loss_cfg.get('start_weight', 3.0),
        start_steps=loss_cfg.get('start_steps', 10),
        transition_weight=loss_cfg.get('transition_weight', 5.0),
        transition_threshold=0.5,
    )


def test_loss():
    """Test loss function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create loss
    loss_fn = ActionLoss(
        gripper_loss_weight=2.0,
        balance_gripper=True,
        start_weight=3.0,
        start_steps=10,
        transition_weight=5.0,
    )

    # Test inputs
    B, T, D = 4, 50, 7
    pred = torch.randn(B, T, D).to(device)
    target = torch.randn(B, T, D).to(device)
    weights = torch.ones(B, T).to(device)

    # Add a gripper transition
    target[:, :25, 6] = -1.0  # open
    target[:, 25:, 6] = 1.0   # closed

    # Compute loss
    losses = loss_fn(pred, target, weights)

    print("Loss values:")
    for k, v in losses.items():
        if v.dim() == 0:
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v.shape}")

    # Test without weights
    losses_no_weights = loss_fn(pred, target)
    print(f"\nLoss without explicit weights: {losses_no_weights['loss'].item():.4f}")

    print("\nLoss test passed!")


if __name__ == "__main__":
    test_loss()
