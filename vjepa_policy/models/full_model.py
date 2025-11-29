"""
Full VJEPA2 Policy Model

Combines V-JEPA 2 encoder + Proprio encoder + Policy head
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from .vjepa2_encoder import VJEPA2Encoder
from .proprio_encoder import ProprioEncoder
from .policy_head import PolicyHead, PolicyHeadSpatial


class VJEPA2Policy(nn.Module):
    """
    Complete V-JEPA 2 Policy model.

    Components:
    - V-JEPA 2 encoder (frozen): Extracts video/image embeddings
    - Proprio encoder (trained): Encodes proprioception history
    - Policy head (trained): Outputs action chunks
    """

    def __init__(
        self,
        # V-JEPA 2 config
        vjepa2_model_path: Optional[str] = "/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name: str = "vjepa2_vitg",
        vjepa2_freeze: bool = True,
        vjepa2_num_frames: int = 16,
        vjepa2_use_attentive_pool: bool = True,
        # Proprio config
        proprio_dim: int = 23,
        proprio_history: int = 5,
        proprio_output_dim: int = 256,
        # Policy head config
        policy_hidden_dim: int = 512,
        policy_n_heads: int = 8,
        policy_n_layers: int = 4,
        policy_n_context_tokens: int = 4,
        # Action config
        action_dim: int = 7,
        chunk_size: int = 50,
        # Device
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # V-JEPA 2 encoder
        self.vjepa2 = VJEPA2Encoder(
            model_path=vjepa2_model_path,
            model_name=vjepa2_model_name,
            freeze=vjepa2_freeze,
            device=device,
            num_frames=vjepa2_num_frames,
            use_attentive_pool=vjepa2_use_attentive_pool,
        )
        vision_dim = self.vjepa2.embed_dim  # 1408 for ViT-Giant

        # Proprio encoder
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=proprio_dim,
            history_len=proprio_history,
            output_dim=proprio_output_dim,
        )

        # Policy head
        self.policy_head = PolicyHead(
            vision_dim=vision_dim,
            proprio_dim=proprio_output_dim,
            hidden_dim=policy_hidden_dim,
            n_heads=policy_n_heads,
            n_layers=policy_n_layers,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_context_tokens=policy_n_context_tokens,
        )

        self.to(device)

    def forward(
        self,
        video: torch.Tensor,
        goal_image: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            video: (B, T, C, H, W) or (B, C, T, H, W) video frames, values in [0, 1]
            goal_image: (B, C, H, W) goal image, values in [0, 1]
            proprio: (B, history_len, proprio_dim) proprio history

        Returns:
            actions: (B, chunk_size, action_dim) predicted actions
        """
        # Encode video
        current_emb = self.vjepa2.encode_video(video)  # (B, vision_dim)

        # Encode goal
        goal_emb = self.vjepa2.encode_image(goal_image)  # (B, vision_dim)

        # Encode proprio
        proprio_emb = self.proprio_encoder(proprio)  # (B, proprio_output_dim)

        # Policy forward
        actions = self.policy_head(current_emb, goal_emb, proprio_emb)

        return actions

    def forward_with_precomputed(
        self,
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pre-computed V-JEPA 2 embeddings.
        Faster for training when embeddings are cached.
        """
        proprio_emb = self.proprio_encoder(proprio)
        actions = self.policy_head(current_emb, goal_emb, proprio_emb)
        return actions

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable parameters (excluding frozen V-JEPA 2 encoder)"""
        params = []

        # V-JEPA 2 pooler (trainable even when encoder is frozen)
        params.extend(self.vjepa2.get_trainable_params())

        # Proprio encoder
        params.extend(self.proprio_encoder.parameters())

        # Policy head
        params.extend(self.policy_head.parameters())

        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component"""
        vjepa2_total = sum(p.numel() for p in self.vjepa2.parameters())
        vjepa2_trainable = sum(p.numel() for p in self.vjepa2.get_trainable_params())
        proprio_params = sum(p.numel() for p in self.proprio_encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())

        return {
            'vjepa2_total': vjepa2_total,
            'vjepa2_trainable': vjepa2_trainable,
            'proprio_encoder': proprio_params,
            'policy_head': policy_params,
            'trainable': trainable_params,
            'total': vjepa2_total + proprio_params + policy_params,
        }


class VJEPA2PolicySpatial(nn.Module):
    """
    V-JEPA 2 Policy model with spatial tokens (no pooling).

    Instead of mean-pooling all patch tokens to 1 embedding,
    this preserves 64 spatial tokens per modality for richer context.

    Components:
    - V-JEPA 2 encoder: Extracts spatial tokens (64, 1408)
    - Proprio encoder: Encodes proprioception history
    - PolicyHeadSpatial: Uses 132 context tokens (64+64+4)
    """

    def __init__(
        self,
        # V-JEPA 2 config
        vjepa2_model_path: Optional[str] = "/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name: str = "vjepa2_vitg",
        vjepa2_freeze: bool = True,
        vjepa2_num_frames: int = 16,
        # Proprio config
        proprio_dim: int = 15,
        proprio_history: int = 5,
        proprio_output_dim: int = 256,
        # Policy head config
        policy_hidden_dim: int = 512,
        policy_n_heads: int = 8,
        policy_n_layers: int = 4,
        n_spatial_tokens: int = 64,
        n_proprio_tokens: int = 4,
        # Action config
        action_dim: int = 7,
        chunk_size: int = 50,
        # Device
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_spatial_tokens = n_spatial_tokens

        # V-JEPA 2 encoder (don't use attentive pool, we'll use spatial encoding)
        self.vjepa2 = VJEPA2Encoder(
            model_path=vjepa2_model_path,
            model_name=vjepa2_model_name,
            freeze=vjepa2_freeze,
            device=device,
            num_frames=vjepa2_num_frames,
            use_attentive_pool=False,  # We use spatial tokens instead
        )
        vision_dim = self.vjepa2.embed_dim  # 1408 for ViT-Giant

        # Proprio encoder
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=proprio_dim,
            history_len=proprio_history,
            output_dim=proprio_output_dim,
        )

        # Policy head for spatial tokens
        self.policy_head = PolicyHeadSpatial(
            vision_dim=vision_dim,
            proprio_dim=proprio_output_dim,
            hidden_dim=policy_hidden_dim,
            n_heads=policy_n_heads,
            n_layers=policy_n_layers,
            action_dim=action_dim,
            chunk_size=chunk_size,
            n_spatial_tokens=n_spatial_tokens,
            n_proprio_tokens=n_proprio_tokens,
        )

        self.to(device)

    def forward(
        self,
        video: torch.Tensor,
        goal_image: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass with spatial tokens.

        Args:
            video: (B, T, C, H, W) or (B, C, T, H, W) video frames
            goal_image: (B, C, H, W) goal image
            proprio: (B, history_len, proprio_dim) proprio history

        Returns:
            actions: (B, chunk_size, action_dim) predicted actions
        """
        # Encode video to spatial tokens: (B, 64, 1408)
        video_tokens = self.vjepa2.encode_video_spatial(video)

        # Encode goal to spatial tokens: (B, 64, 1408)
        goal_tokens = self.vjepa2.encode_image_spatial(goal_image)

        # Encode proprio
        proprio_emb = self.proprio_encoder(proprio)

        # Policy forward with spatial tokens
        actions = self.policy_head(video_tokens, goal_tokens, proprio_emb)

        return actions

    def forward_with_precomputed(
        self,
        video_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pre-computed spatial tokens.

        Args:
            video_tokens: (B, 64, 1408) precomputed video spatial tokens
            goal_tokens: (B, 64, 1408) precomputed goal spatial tokens
            proprio: (B, history_len, proprio_dim) proprio history

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        proprio_emb = self.proprio_encoder(proprio)
        actions = self.policy_head(video_tokens, goal_tokens, proprio_emb)
        return actions

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable parameters"""
        params = []
        params.extend(self.vjepa2.get_trainable_params())
        params.extend(self.proprio_encoder.parameters())
        params.extend(self.policy_head.parameters())
        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component"""
        vjepa2_total = sum(p.numel() for p in self.vjepa2.parameters())
        vjepa2_trainable = sum(p.numel() for p in self.vjepa2.get_trainable_params())
        proprio_params = sum(p.numel() for p in self.proprio_encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())

        return {
            'vjepa2_total': vjepa2_total,
            'vjepa2_trainable': vjepa2_trainable,
            'proprio_encoder': proprio_params,
            'policy_head': policy_params,
            'trainable': trainable_params,
            'total': vjepa2_total + proprio_params + policy_params,
        }


def test_full_model():
    """Test full model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model with local weights
    print("\nCreating VJEPA2Policy...")
    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        vjepa2_freeze=True,
        policy_hidden_dim=512,
        policy_n_context_tokens=4,
        device=device,
    )

    # Test inputs
    B = 2
    video = torch.rand(B, 16, 3, 256, 256).to(device)  # (B, T, C, H, W)
    goal = torch.rand(B, 3, 256, 256).to(device)
    proprio = torch.rand(B, 5, 23).to(device)

    # Forward pass
    print("\nRunning forward pass...")
    actions = model(video, goal, proprio)

    print(f"Video shape: {video.shape}")
    print(f"Goal shape: {goal.shape}")
    print(f"Proprio shape: {proprio.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Parameter counts
    counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count / 1e6:.2f}M")

    # Test with precomputed embeddings
    print("\nTesting forward_with_precomputed...")
    with torch.no_grad():
        current_emb = model.vjepa2.encode_video(video)
        goal_emb = model.vjepa2.encode_image(goal)
    actions_precomputed = model.forward_with_precomputed(current_emb, goal_emb, proprio)
    print(f"Precomputed actions shape: {actions_precomputed.shape}")

    print("\n✅ Full model test passed!")


if __name__ == "__main__":
    test_full_model()
