"""
ACT-Style Policy Head

Transformer-based policy that outputs action chunks.
Input: V-JEPA 2 embedding + proprio + goal embedding
Output: Chunk of 50 actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for action queries"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PolicyHead(nn.Module):
    """
    ACT-style transformer policy head.
    
    Architecture:
    1. Fuse vision + proprio + goal into context
    2. Learnable action queries (one per timestep)
    3. Transformer decoder: queries attend to context
    4. MLP head outputs actions
    """
    
    def __init__(
        self,
        vision_dim: int = 1280,      # V-JEPA 2 embedding dim
        proprio_dim: int = 256,       # Encoded proprio dim
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        action_dim: int = 7,
        chunk_size: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Input projections
        # Context: current_vision + goal_vision + proprio
        context_dim = vision_dim + vision_dim + proprio_dim
        
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Learnable action queries
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim))
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=chunk_size)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Action output head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bound actions to [-1, 1]
        )
    
    def forward(
        self,
        current_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        proprio_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_emb: (B, vision_dim) V-JEPA 2 video embedding
            goal_emb: (B, vision_dim) V-JEPA 2 goal image embedding
            proprio_emb: (B, proprio_dim) encoded proprioception
        
        Returns:
            actions: (B, chunk_size, action_dim) action chunk
        """
        B = current_emb.shape[0]
        
        # 1. Create context
        context = torch.cat([current_emb, goal_emb, proprio_emb], dim=-1)  # (B, context_dim)
        context = self.context_proj(context)  # (B, hidden_dim)
        context = context.unsqueeze(1)  # (B, 1, hidden_dim) - single context token
        
        # 2. Prepare action queries
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, hidden_dim)
        queries = self.pos_encoding(queries)  # Add positional encoding
        
        # 3. Transformer decoder
        # Queries attend to context
        action_features = self.transformer(
            tgt=queries,      # (B, chunk_size, hidden_dim)
            memory=context,   # (B, 1, hidden_dim)
        )
        
        # 4. Predict actions
        actions = self.action_head(action_features)  # (B, chunk_size, action_dim)
        
        return actions


def test_policy_head():
    """Test policy head"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = PolicyHead(
        vision_dim=1280,
        proprio_dim=256,
        hidden_dim=256,
        action_dim=7,
        chunk_size=50,
    ).to(device)
    
    # Test inputs
    B = 4
    current_emb = torch.rand(B, 1280).to(device)
    goal_emb = torch.rand(B, 1280).to(device)
    proprio_emb = torch.rand(B, 256).to(device)
    
    # Forward pass
    actions = policy(current_emb, goal_emb, proprio_emb)
    
    print(f"Current emb shape: {current_emb.shape}")
    print(f"Goal emb shape: {goal_emb.shape}")
    print(f"Proprio emb shape: {proprio_emb.shape}")
    print(f"Actions shape: {actions.shape}")  # Should be (4, 50, 7)
    
    # Check action bounds
    assert actions.min() >= -1.0 and actions.max() <= 1.0, "Actions out of bounds!"
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy head parameters: {n_params / 1e6:.2f}M")
    
    print("Policy head test passed!")


if __name__ == "__main__":
    test_policy_head()