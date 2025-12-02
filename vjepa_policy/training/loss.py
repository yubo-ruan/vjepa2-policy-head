"""
Unified Loss Function for V-JEPA 2 Policy Training.

This module provides the ActionLoss class that combines multiple loss components
and weighting strategies to train the policy effectively.

Key Features:
    - Per-timestep weighting from dataset (start and transition emphasis)
    - Gripper class balancing (handles imbalanced OPEN vs CLOSE samples)
    - Extra gripper dimension weight (gripper timing is critical)
    - Temporal weighting (computed dynamically if not provided)

Loss Computation:
    1. Position/rotation loss: L1 loss on dims 0-5 (OSC position + rotation deltas)
    2. Gripper loss: L1 loss on dim 6, with class balancing and extra weight
    3. Temporal weighting: Higher weights on start timesteps (initial positioning)
       and gripper transitions (grasp/release moments)

Example:
    >>> loss_fn = ActionLoss(gripper_loss_weight=2.0, start_weight=3.0)
    >>> pred = model(video_tokens, goal_tokens, proprio)
    >>> losses = loss_fn(pred, target_actions, weights)
    >>> losses['loss'].backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ActionLoss(nn.Module):
    """
    Unified action loss with configurable weighting for robot manipulation.

    This loss function addresses several challenges in imitation learning:

    1. **Gripper Timing**: The gripper action (open/close) must be precisely
       timed for successful manipulation. This is handled via:
       - Extra weight on gripper dimension (gripper_loss_weight)
       - Class balancing for OPEN vs CLOSE samples
       - Higher weight around transition timesteps

    2. **Initial Actions**: The first few actions are critical for positioning.
       Errors early compound throughout the trajectory. Handled via start_weight.

    3. **Transition Moments**: Gripper state changes (open→close, close→open)
       are key moments that often determine task success. Handled via
       transition_weight applied to timesteps around detected transitions.

    Args:
        gripper_loss_weight: Multiplier for gripper dimension loss (default: 2.0)
        balance_gripper: Whether to balance OPEN vs CLOSE samples (default: True)
        gripper_dim: Index of gripper in action vector (default: 6, i.e., last dim)
        start_weight: Weight multiplier for first N timesteps (default: 3.0)
        start_steps: Number of initial timesteps to upweight (default: 10)
        transition_weight: Weight for timesteps around gripper transitions (default: 5.0)
        transition_threshold: Min gripper value change to detect transition (default: 0.5)
        separate_gripper_head: If True, gripper is a logit (use BCE loss) (default: False)
        focal_gamma: Focal loss gamma for gripper classification (default: 2.0)

    Returns:
        Dictionary containing:
            - loss: Weighted total loss (for backprop)
            - loss_unweighted: Unweighted loss (for comparison)
            - loss_per_dim: Per-action-dimension loss
            - pos_loss: Position/rotation component
            - gripper_loss: Gripper component
            - weights_mean: Average weight applied
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
        separate_gripper_head: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.gripper_loss_weight = gripper_loss_weight
        self.balance_gripper = balance_gripper
        self.gripper_dim = gripper_dim
        self.start_weight = start_weight
        self.start_steps = start_steps
        self.transition_weight = transition_weight
        self.transition_threshold = transition_threshold
        self.separate_gripper_head = separate_gripper_head
        self.focal_gamma = focal_gamma

    def compute_temporal_weights(
        self,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-timestep weights based on position and transitions.

        Creates a weight tensor that emphasizes:
        1. Initial timesteps (start_weight for first start_steps)
        2. Gripper transition moments (transition_weight around changes)

        Args:
            target_actions: (B, T, D) target actions tensor

        Returns:
            weights: (B, T) temporal weight tensor
        """
        B, T, D = target_actions.shape
        device = target_actions.device

        # Start with base weights, upweight initial timesteps
        weights = torch.ones(B, T, device=device)
        weights[:, :self.start_steps] = self.start_weight

        # Detect and upweight gripper transitions
        if self.gripper_dim < D:
            gripper = target_actions[:, :, self.gripper_dim]
            # Compute absolute change in gripper value
            gripper_diff = torch.zeros_like(gripper)
            gripper_diff[:, 1:] = torch.abs(gripper[:, 1:] - gripper[:, :-1])
            # Mark transitions where change exceeds threshold
            transitions = gripper_diff > self.transition_threshold
            # Add transition weight (additive, not replace)
            weights = weights + transitions.float() * (self.transition_weight - 1.0)

        return weights

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal loss for binary classification.

        Args:
            logits: Raw logits (B, T) or (B, T, 1)
            targets: Binary targets 0 or 1 (B, T)
            gamma: Focusing parameter (higher = more focus on hard samples)

        Returns:
            loss: (B, T) per-sample loss
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute pt (probability of correct class)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Apply focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** gamma

        return focal_weight * bce_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted action loss.

        The loss computation follows these steps:
        1. Split actions into position/rotation (dims 0-5) and gripper (dim 6)
        2. Compute L1 loss for position, BCE/focal loss for gripper (if separate head)
        3. Apply gripper class balancing if enabled
        4. Combine with gripper weight multiplier
        5. Apply temporal weights (from dataset or computed)

        Args:
            pred: (B, T, D) predicted actions from policy
            target: (B, T, D) ground truth actions from demonstrations
            weights: (B, T) optional per-timestep weights from dataset

        Returns:
            Dictionary with loss values (see class docstring)
        """
        B, T, D = pred.shape
        device = pred.device

        # Split position/rotation (dims 0-5) and gripper (dim 6)
        pred_pos = pred[:, :, :self.gripper_dim]
        pred_grip = pred[:, :, self.gripper_dim]
        target_pos = target[:, :, :self.gripper_dim]
        target_grip = target[:, :, self.gripper_dim]

        # Position/rotation loss (L1, averaged over action dims)
        pos_loss = F.l1_loss(pred_pos, target_pos, reduction='none').mean(dim=-1)  # (B, T)

        # Gripper loss
        if self.separate_gripper_head:
            # Gripper is a logit - use focal loss with BCE
            # Convert target from [-1, 1] to [0, 1] for BCE
            # -1 (OPEN) -> 0, +1 (CLOSE) -> 1
            target_grip_binary = (target_grip > 0).float()

            # Focal loss for gripper classification
            grip_loss = self.focal_loss(pred_grip, target_grip_binary, gamma=self.focal_gamma)  # (B, T)

            # Compute gripper accuracy for logging
            pred_grip_binary = (torch.sigmoid(pred_grip) > 0.5).float()
            gripper_acc = (pred_grip_binary == target_grip_binary).float().mean()
        else:
            # Original L1 loss for gripper
            grip_loss = F.l1_loss(pred_grip, target_grip, reduction='none')  # (B, T)
            gripper_acc = torch.tensor(0.0, device=device)

            # Balance gripper classes (OPEN vs CLOSE are often imbalanced)
            # This prevents the model from always predicting the majority class
            if self.balance_gripper:
                n_open = (target_grip < 0).float().sum() + 1e-6
                n_close = (target_grip > 0).float().sum() + 1e-6
                total = n_open + n_close

                # Inverse frequency weighting
                weight_open = total / (2 * n_open)
                weight_close = total / (2 * n_close)

                grip_weights = torch.where(target_grip < 0, weight_open, weight_close)
                grip_loss = grip_loss * grip_weights

        # Combine position and gripper losses
        combined = pos_loss + self.gripper_loss_weight * grip_loss  # (B, T)

        # Get temporal weights (from dataset or computed dynamically)
        if weights is not None:
            temporal_weights = weights.to(device)
        else:
            temporal_weights = self.compute_temporal_weights(target)

        # Apply temporal weights with normalization
        weighted_loss = (combined * temporal_weights).sum() / (temporal_weights.sum() + 1e-6)

        # Unweighted loss for logging/comparison
        unweighted_loss = combined.mean()

        # Per-dimension L1 loss for detailed analysis
        l1_loss = torch.abs(pred - target)
        loss_per_dim = l1_loss.mean(dim=(0, 1))

        result = {
            'loss': weighted_loss,
            'loss_unweighted': unweighted_loss,
            'loss_per_dim': loss_per_dim,
            'pos_loss': pos_loss.mean(),
            'gripper_loss': grip_loss.mean(),
            'weights_mean': temporal_weights.mean(),
        }

        # Add gripper accuracy if using separate head
        if self.separate_gripper_head:
            result['gripper_acc'] = gripper_acc

        return result

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Allow calling without .forward()"""
        return self.forward(pred, target, weights)


def create_loss(config: dict) -> ActionLoss:
    """
    Create ActionLoss instance from configuration dictionary.

    Args:
        config: Configuration dict with optional 'loss' section containing:
            - gripper_loss_weight (default: 2.0)
            - balance_gripper (default: True)
            - gripper_dim (default: 6)
            - start_weight (default: 3.0)
            - start_steps (default: 10)
            - transition_weight (default: 5.0)
            - separate_gripper_head (default: False) - use BCE/focal loss for gripper
            - focal_gamma (default: 2.0) - focal loss gamma parameter

    Returns:
        Configured ActionLoss instance
    """
    loss_cfg = config.get('loss', {})
    model_cfg = config.get('model', {})

    # Check if separate_gripper_head is enabled in model config
    separate_gripper_head = model_cfg.get('separate_gripper_head', False)

    return ActionLoss(
        gripper_loss_weight=loss_cfg.get('gripper_loss_weight', 2.0),
        balance_gripper=loss_cfg.get('balance_gripper', True),
        gripper_dim=loss_cfg.get('gripper_dim', 6),
        start_weight=loss_cfg.get('start_weight', 3.0),
        start_steps=loss_cfg.get('start_steps', 10),
        transition_weight=loss_cfg.get('transition_weight', 5.0),
        transition_threshold=0.5,
        separate_gripper_head=separate_gripper_head,
        focal_gamma=loss_cfg.get('focal_gamma', 2.0),
    )


def test_loss():
    """Test loss function with synthetic data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create loss function
    loss_fn = ActionLoss(
        gripper_loss_weight=2.0,
        balance_gripper=True,
        start_weight=3.0,
        start_steps=10,
        transition_weight=5.0,
    )

    # Create test inputs
    B, T, D = 4, 50, 7
    pred = torch.randn(B, T, D).to(device)
    target = torch.randn(B, T, D).to(device)
    weights = torch.ones(B, T).to(device)

    # Add a gripper transition for testing
    target[:, :25, 6] = -1.0  # gripper open
    target[:, 25:, 6] = 1.0   # gripper closed

    # Compute loss
    losses = loss_fn(pred, target, weights)

    print("Loss values:")
    for k, v in losses.items():
        if v.dim() == 0:
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v.shape}")

    # Test without explicit weights (uses computed weights)
    losses_no_weights = loss_fn(pred, target)
    print(f"\nLoss without explicit weights: {losses_no_weights['loss'].item():.4f}")

    print("\nLoss test passed!")


if __name__ == "__main__":
    test_loss()
