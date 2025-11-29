"""
Proprioception Encoder

Encodes robot proprioceptive state (joint positions, velocities, gripper state, etc.)
"""

import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):
    """
    Encodes proprioception history into a fixed-size embedding.
    
    Input: (B, history_len, proprio_dim) - sequence of proprio states
    Output: (B, output_dim) - encoded proprioception
    """
    
    def __init__(
        self,
        proprio_dim: int = 23,
        history_len: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.proprio_dim = proprio_dim
        self.history_len = history_len
        
        # Flatten and encode
        input_dim = proprio_dim * history_len
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: (B, history_len, proprio_dim) or (B, history_len * proprio_dim)
        
        Returns:
            encoding: (B, output_dim)
        """
        if proprio.dim() == 3:
            B, H, D = proprio.shape
            proprio = proprio.view(B, -1)  # Flatten
        
        return self.encoder(proprio)


def test_proprio_encoder():
    """Test proprio encoder"""
    encoder = ProprioEncoder(proprio_dim=23, history_len=5, output_dim=256)
    
    # Test input
    proprio = torch.rand(4, 5, 23)  # (batch=4, history=5, dim=23)
    output = encoder(proprio)
    
    print(f"Input shape: {proprio.shape}")
    print(f"Output shape: {output.shape}")  # Should be (4, 256)
    
    assert output.shape == (4, 256), "Output shape mismatch!"
    print("Proprio encoder test passed!")


if __name__ == "__main__":
    test_proprio_encoder()