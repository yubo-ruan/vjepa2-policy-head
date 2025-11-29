"""
Full VJEPA2 Policy Model

Combines V-JEPA 2 encoder + Proprio encoder + Policy head
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .vjepa2_encoder import VJEPA2Encoder
from .proprio_encoder import ProprioEncoder
from .policy_head import PolicyHead


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
        vjepa2_model: str = "facebook/vjepa2-vitg-fpc64-384",
        vjepa2_freeze: bool = True,
        proprio_dim: int = 23,
        proprio_history: int = 5,
        proprio_output_dim: int = 256,
        policy_hidden_dim: int = 256,
        policy_n_heads: int = 8,
        policy_n_layers: int = 4,
        action_dim: int = 7,
        chunk_size: int = 50,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = device
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        
        # V-JEPA 2 encoder
        self.vjepa2 = VJEPA2Encoder(
            model_name=vjepa2_model,
            freeze=vjepa2_freeze,
            device=device,
        )
        vision_dim = self.vjepa2.embed_dim
        
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
            video: (B, T, C, H, W) video frames, values in [0, 1]
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
    
    def get_trainable_params(self):
        """Get only trainable parameters (excluding frozen V-JEPA 2)"""
        params = []
        params.extend(self.proprio_encoder.parameters())
        params.extend(self.policy_head.parameters())
        return params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component"""
        vjepa2_params = sum(p.numel() for p in self.vjepa2.parameters())
        proprio_params = sum(p.numel() for p in self.proprio_encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        
        return {
            'vjepa2': vjepa2_params,
            'proprio_encoder': proprio_params,
            'policy_head': policy_params,
            'trainable': trainable_params,
            'total': vjepa2_params + proprio_params + policy_params,
        }


def test_full_model():
    """Test full model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use smaller model for testing
    model = VJEPA2Policy(
        vjepa2_model="facebook/vjepa2-vitl-fpc64-256",  # Smaller
        device=device,
    )
    
    # Test inputs
    B = 2
    video = torch.rand(B, 16, 3, 256, 256).to(device)
    goal = torch.rand(B, 3, 256, 256).to(device)
    proprio = torch.rand(B, 5, 23).to(device)
    
    # Forward pass
    actions = model(video, goal, proprio)
    
    print(f"Video shape: {video.shape}")
    print(f"Goal shape: {goal.shape}")
    print(f"Proprio shape: {proprio.shape}")
    print(f"Actions shape: {actions.shape}")
    
    # Parameter counts
    counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count / 1e6:.2f}M")
    
    print("\nFull model test passed!")


if __name__ == "__main__":
    test_full_model()