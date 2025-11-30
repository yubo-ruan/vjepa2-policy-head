"""
V-JEPA 2 Policy Model - Spatial Tokens Only.

Single policy head architecture using 64 spatial tokens per modality (8x8 grid).
No mean-pooling variants - this is the only architecture.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for action queries."""

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


class ProprioEncoder(nn.Module):
    """
    Encode proprioception history into a fixed-size embedding.

    Takes flattened proprio history and produces a single embedding.
    """

    def __init__(
        self,
        proprio_dim: int = 15,
        history_len: int = 5,
        output_dim: int = 256,
    ):
        super().__init__()

        input_dim = proprio_dim * history_len

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: (B, history_len, proprio_dim) or (B, history_len * proprio_dim)

        Returns:
            embedding: (B, output_dim)
        """
        if proprio.dim() == 3:
            proprio = proprio.flatten(1)  # (B, history_len * proprio_dim)
        return self.encoder(proprio)


class PolicyHead(nn.Module):
    """
    Transformer decoder policy head for spatial tokens.

    Input: 64 video tokens + 64 goal tokens + proprio embedding
    Output: Action chunk (chunk_size x action_dim)

    Args:
        embed_dim: V-JEPA 2 embedding dimension (1408)
        hidden_dim: Policy head hidden dimension
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        num_spatial_tokens: Number of spatial tokens per modality (64)
        chunk_size: Action chunk size
        action_dim: Action dimensions
        proprio_dim: Encoded proprioception dimension
        n_proprio_tokens: Number of proprio context tokens
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_spatial_tokens: int = 64,
        chunk_size: int = 50,
        action_dim: int = 7,
        proprio_dim: int = 256,
        n_proprio_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_spatial_tokens = num_spatial_tokens
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_proprio_tokens = n_proprio_tokens

        # Project spatial tokens from embed_dim to hidden_dim
        self.video_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.goal_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Proprio projection: (B, proprio_dim) -> (B, n_proprio_tokens, hidden_dim)
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim * n_proprio_tokens),
            nn.LayerNorm(hidden_dim * n_proprio_tokens),
            nn.GELU(),
        )

        # Learnable modality embeddings (added to each token)
        self.video_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.goal_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.proprio_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable spatial position embeddings for video and goal
        self.spatial_pos = nn.Parameter(torch.randn(1, num_spatial_tokens, hidden_dim) * 0.02)

        # Learnable action queries
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=chunk_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
            activation='gelu',
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action output head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

        # Learnable output scale - starts small to prevent tanh saturation
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        video_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        proprio_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video_tokens: (B, 64, embed_dim) spatial video features
            goal_tokens: (B, 64, embed_dim) spatial goal features
            proprio_emb: (B, proprio_dim) encoded proprioception

        Returns:
            actions: (B, chunk_size, action_dim) predicted actions
        """
        B = video_tokens.shape[0]

        # Project spatial tokens: (B, 64, embed_dim) -> (B, 64, hidden_dim)
        video_proj = self.video_proj(video_tokens) + self.video_emb + self.spatial_pos
        goal_proj = self.goal_proj(goal_tokens) + self.goal_emb + self.spatial_pos

        # Project proprio: (B, proprio_dim) -> (B, n_proprio_tokens, hidden_dim)
        proprio_proj = self.proprio_proj(proprio_emb).view(B, self.n_proprio_tokens, self.hidden_dim)
        proprio_proj = proprio_proj + self.proprio_emb

        # Concatenate all context tokens: 64 + 64 + n_proprio = 132 tokens (with n_proprio=4)
        context = torch.cat([video_proj, goal_proj, proprio_proj], dim=1)

        # Prepare action queries
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        # Transformer decoder
        action_features = self.transformer(tgt=queries, memory=context)

        # Predict actions with scaled tanh to prevent saturation
        raw_actions = self.action_head(action_features)
        actions = torch.tanh(raw_actions * self.output_scale)

        return actions


class VJEPA2Policy(nn.Module):
    """
    Full V-JEPA 2 Policy model with spatial tokens.

    Combines:
    - V-JEPA 2 encoder (frozen, for live inference)
    - Proprio encoder (trained)
    - Policy head (trained)

    For training with precomputed embeddings, use forward_with_precomputed().
    For evaluation with live video input, use forward().
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
        embed_dim: int = 1408,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_spatial_tokens: int = 64,
        n_proprio_tokens: int = 4,
        # Action config
        action_dim: int = 7,
        chunk_size: int = 50,
        dropout: float = 0.1,
        # Device
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_spatial_tokens = num_spatial_tokens

        # V-JEPA 2 encoder (lazy load to avoid import issues during testing)
        self._vjepa2 = None
        self._vjepa2_config = {
            'model_path': vjepa2_model_path,
            'model_name': vjepa2_model_name,
            'freeze': vjepa2_freeze,
            'num_frames': vjepa2_num_frames,
        }

        # Proprio encoder
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=proprio_dim,
            history_len=proprio_history,
            output_dim=proprio_output_dim,
        )

        # Policy head for spatial tokens
        self.policy_head = PolicyHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_spatial_tokens=num_spatial_tokens,
            chunk_size=chunk_size,
            action_dim=action_dim,
            proprio_dim=proprio_output_dim,
            n_proprio_tokens=n_proprio_tokens,
            dropout=dropout,
        )

        self.to(device)

    @property
    def vjepa2(self):
        """Lazy load V-JEPA 2 encoder."""
        if self._vjepa2 is None:
            from .vjepa2_encoder import VJEPA2Encoder
            self._vjepa2 = VJEPA2Encoder(
                model_path=self._vjepa2_config['model_path'],
                model_name=self._vjepa2_config['model_name'],
                freeze=self._vjepa2_config['freeze'],
                device=self.device,
                num_frames=self._vjepa2_config['num_frames'],
                use_attentive_pool=False,  # We use spatial tokens
            )
        return self._vjepa2

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to spatial tokens.

        Args:
            video: (B, T, C, H, W) or (B, C, T, H, W) video frames

        Returns:
            tokens: (B, 64, 1408) spatial tokens
        """
        return self.vjepa2.encode_video_spatial(video)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to spatial tokens.

        Args:
            image: (B, C, H, W) image

        Returns:
            tokens: (B, 64, 1408) spatial tokens
        """
        return self.vjepa2.encode_image_spatial(image)

    def forward(
        self,
        video: torch.Tensor,
        goal_image: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass with video encoding.

        Args:
            video: (B, T, C, H, W) or (B, C, T, H, W) video frames
            goal_image: (B, C, H, W) goal image
            proprio: (B, history_len, proprio_dim) proprio history

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        # Encode video to spatial tokens: (B, 64, 1408)
        video_tokens = self.encode_video(video)

        # Encode goal to spatial tokens: (B, 64, 1408)
        goal_tokens = self.encode_image(goal_image)

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
            proprio: (B, history_len, proprio_dim) or (B, proprio_dim * history) proprio

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        proprio_emb = self.proprio_encoder(proprio)
        actions = self.policy_head(video_tokens, goal_tokens, proprio_emb)
        return actions

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get only trainable parameters (excluding frozen V-JEPA 2 encoder)."""
        params = []

        # V-JEPA 2 pooler (if loaded and has trainable params)
        if self._vjepa2 is not None:
            params.extend(self._vjepa2.get_trainable_params())

        # Proprio encoder
        params.extend(self.proprio_encoder.parameters())

        # Policy head
        params.extend(self.policy_head.parameters())

        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        proprio_params = sum(p.numel() for p in self.proprio_encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())

        counts = {
            'proprio_encoder': proprio_params,
            'policy_head': policy_params,
            'trainable': trainable_params,
        }

        if self._vjepa2 is not None:
            vjepa2_total = sum(p.numel() for p in self._vjepa2.parameters())
            vjepa2_trainable = sum(p.numel() for p in self._vjepa2.get_trainable_params())
            counts['vjepa2_total'] = vjepa2_total
            counts['vjepa2_trainable'] = vjepa2_trainable
            counts['total'] = vjepa2_total + proprio_params + policy_params

        return counts


def create_policy_head(config: dict) -> PolicyHead:
    """Create policy head from config."""
    model_cfg = config['model']
    return PolicyHead(
        embed_dim=model_cfg.get('embed_dim', 1408),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        num_layers=model_cfg.get('num_layers', 4),
        num_heads=model_cfg.get('num_heads', 8),
        num_spatial_tokens=model_cfg.get('num_spatial_tokens', 64),
        chunk_size=model_cfg.get('chunk_size', 50),
        action_dim=model_cfg.get('action_dim', 7),
        proprio_dim=256,  # Fixed proprio encoder output dim
        n_proprio_tokens=4,
        dropout=model_cfg.get('dropout', 0.1),
    )


def create_model(config: dict, device: str = "cuda") -> VJEPA2Policy:
    """Create full model from config."""
    model_cfg = config['model']
    encoder_cfg = config.get('encoder', {})

    return VJEPA2Policy(
        vjepa2_model_path=encoder_cfg.get('model_path', "/workspace/models/vjepa2-ac-vitg.pt"),
        vjepa2_model_name=encoder_cfg.get('model_name', "vjepa2_vitg"),
        vjepa2_freeze=encoder_cfg.get('freeze', True),
        vjepa2_num_frames=encoder_cfg.get('num_frames', 16),
        proprio_dim=model_cfg.get('proprio_dim', 15),
        proprio_history=model_cfg.get('proprio_history', 5),
        proprio_output_dim=256,
        embed_dim=model_cfg.get('embed_dim', 1408),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        num_heads=model_cfg.get('num_heads', 8),
        num_layers=model_cfg.get('num_layers', 4),
        num_spatial_tokens=model_cfg.get('num_spatial_tokens', 64),
        n_proprio_tokens=4,
        action_dim=model_cfg.get('action_dim', 7),
        chunk_size=model_cfg.get('chunk_size', 50),
        dropout=model_cfg.get('dropout', 0.1),
        device=device,
    )


def test_policy_head():
    """Test policy head."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # Create policy head
    policy = PolicyHead(
        embed_dim=1408,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        num_spatial_tokens=64,
        chunk_size=50,
        action_dim=7,
    ).to(device)

    # Test inputs - spatial tokens
    B = 4
    video_tokens = torch.rand(B, 64, 1408).to(device)
    goal_tokens = torch.rand(B, 64, 1408).to(device)
    proprio_emb = torch.rand(B, 256).to(device)

    # Forward pass
    actions = policy(video_tokens, goal_tokens, proprio_emb)

    print(f"Video tokens shape: {video_tokens.shape}")
    print(f"Goal tokens shape: {goal_tokens.shape}")
    print(f"Proprio emb shape: {proprio_emb.shape}")
    print(f"Actions shape: {actions.shape}")

    # Check action bounds
    assert actions.min() >= -1.0 and actions.max() <= 1.0, "Actions out of bounds!"
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    print("\nPolicy head test passed!")


if __name__ == "__main__":
    test_policy_head()
