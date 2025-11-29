"""
ACT-Style Policy Head

Transformer-based policy that outputs action chunks.
Input: V-JEPA 2 embedding + proprio + goal embedding
Output: Chunk of 50 actions

Improved architecture:
- Multiple context tokens (not single compressed token)
- Separate projections for vision, goal, proprio
- Cross-attention between action queries and context
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
    ACT-style transformer policy head with improved architecture.

    Key improvements over naive single-token context:
    1. Separate projection for each input modality
    2. Multiple context tokens (current + goal + proprio)
    3. Learnable modality embeddings
    4. Optional: can accept patch tokens directly for richer context
    """

    def __init__(
        self,
        vision_dim: int = 1408,       # V-JEPA 2 ViT-Giant embedding dim
        proprio_dim: int = 256,        # Encoded proprio dim
        hidden_dim: int = 512,         # Transformer hidden dim
        n_heads: int = 8,
        n_layers: int = 4,
        action_dim: int = 7,
        chunk_size: int = 50,
        dropout: float = 0.1,
        n_context_tokens: int = 4,     # Number of context tokens per modality
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_context_tokens = n_context_tokens

        # Separate projections for each modality -> multiple tokens each
        self.current_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim * n_context_tokens),
            nn.LayerNorm(hidden_dim * n_context_tokens),
            nn.GELU(),
        )

        self.goal_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim * n_context_tokens),
            nn.LayerNorm(hidden_dim * n_context_tokens),
            nn.GELU(),
        )

        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim * n_context_tokens),
            nn.LayerNorm(hidden_dim * n_context_tokens),
            nn.GELU(),
        )

        # Learnable modality embeddings
        self.current_emb = nn.Parameter(torch.randn(1, n_context_tokens, hidden_dim) * 0.02)
        self.goal_emb = nn.Parameter(torch.randn(1, n_context_tokens, hidden_dim) * 0.02)
        self.proprio_emb = nn.Parameter(torch.randn(1, n_context_tokens, hidden_dim) * 0.02)

        # Learnable action queries
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=chunk_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Action output head - use smaller final layer to prevent tanh saturation
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
        """Initialize weights with small values for stable training"""
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

        # 1. Project each modality to multiple context tokens
        # (B, vision_dim) -> (B, n_context_tokens, hidden_dim)
        current_tokens = self.current_proj(current_emb).view(B, self.n_context_tokens, self.hidden_dim)
        goal_tokens = self.goal_proj(goal_emb).view(B, self.n_context_tokens, self.hidden_dim)
        proprio_tokens = self.proprio_proj(proprio_emb).view(B, self.n_context_tokens, self.hidden_dim)

        # 2. Add modality embeddings
        current_tokens = current_tokens + self.current_emb
        goal_tokens = goal_tokens + self.goal_emb
        proprio_tokens = proprio_tokens + self.proprio_emb

        # 3. Concatenate all context tokens
        # Total context: 3 * n_context_tokens tokens
        context = torch.cat([current_tokens, goal_tokens, proprio_tokens], dim=1)
        # (B, 3 * n_context_tokens, hidden_dim)

        # 4. Prepare action queries
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, hidden_dim)
        queries = self.pos_encoding(queries)  # Add positional encoding

        # 5. Transformer decoder: queries attend to context
        action_features = self.transformer(
            tgt=queries,      # (B, chunk_size, hidden_dim)
            memory=context,   # (B, 3 * n_context_tokens, hidden_dim)
        )

        # 6. Predict actions with scaled tanh to prevent saturation
        raw_actions = self.action_head(action_features)  # (B, chunk_size, action_dim)
        actions = torch.tanh(raw_actions * self.output_scale)

        return actions


class PolicyHeadSpatial(nn.Module):
    """
    Policy head for spatial token input (64 tokens per modality).

    Instead of mean-pooled single embedding per modality, this uses
    64 spatial tokens from AdaptiveAvgPool2d((8,8)) downsampling.

    Context tokens: 64 (video) + 64 (goal) + 4 (proprio) = 132 tokens
    """

    def __init__(
        self,
        vision_dim: int = 1408,       # V-JEPA 2 ViT-Giant embedding dim
        proprio_dim: int = 256,        # Encoded proprio dim
        hidden_dim: int = 512,         # Transformer hidden dim
        n_heads: int = 8,
        n_layers: int = 4,
        action_dim: int = 7,
        chunk_size: int = 50,
        dropout: float = 0.1,
        n_spatial_tokens: int = 64,    # Number of spatial tokens (8x8)
        n_proprio_tokens: int = 4,     # Number of proprio tokens
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_spatial_tokens = n_spatial_tokens
        self.n_proprio_tokens = n_proprio_tokens

        # Project spatial tokens from vision_dim to hidden_dim
        self.video_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.goal_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
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
        self.spatial_pos = nn.Parameter(torch.randn(1, n_spatial_tokens, hidden_dim) * 0.02)

        # Learnable action queries
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=chunk_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Action output head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

        # Learnable output scale
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
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
        Args:
            video_tokens: (B, 64, vision_dim) spatial video tokens
            goal_tokens: (B, 64, vision_dim) spatial goal tokens
            proprio_emb: (B, proprio_dim) encoded proprioception

        Returns:
            actions: (B, chunk_size, action_dim) action chunk
        """
        B = video_tokens.shape[0]

        # 1. Project spatial tokens: (B, 64, vision_dim) -> (B, 64, hidden_dim)
        video_proj = self.video_proj(video_tokens) + self.video_emb + self.spatial_pos
        goal_proj = self.goal_proj(goal_tokens) + self.goal_emb + self.spatial_pos

        # 2. Project proprio: (B, proprio_dim) -> (B, n_proprio_tokens, hidden_dim)
        proprio_proj = self.proprio_proj(proprio_emb).view(B, self.n_proprio_tokens, self.hidden_dim)
        proprio_proj = proprio_proj + self.proprio_emb

        # 3. Concatenate all context tokens: 64 + 64 + 4 = 132 tokens
        context = torch.cat([video_proj, goal_proj, proprio_proj], dim=1)

        # 4. Prepare action queries
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        # 5. Transformer decoder
        action_features = self.transformer(tgt=queries, memory=context)

        # 6. Predict actions with scaled tanh
        raw_actions = self.action_head(action_features)
        actions = torch.tanh(raw_actions * self.output_scale)

        return actions


class PolicyHeadWithPatches(nn.Module):
    """
    Alternative policy head that can use patch tokens directly.
    More expressive but higher memory usage.
    """

    def __init__(
        self,
        vision_dim: int = 1408,
        proprio_dim: int = 256,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        action_dim: int = 7,
        chunk_size: int = 50,
        dropout: float = 0.1,
        max_patches: int = 256,  # Max number of patch tokens to use
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.max_patches = max_patches

        # Project patch tokens to hidden dim
        self.patch_proj = nn.Linear(vision_dim, hidden_dim)

        # Proprio projection
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learnable tokens for current vs goal distinction
        self.current_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.goal_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.proprio_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Action queries
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=chunk_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        current_patches: torch.Tensor,
        goal_patches: torch.Tensor,
        proprio_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_patches: (B, N, vision_dim) V-JEPA 2 patch tokens
            goal_patches: (B, M, vision_dim) goal patch tokens
            proprio_emb: (B, proprio_dim) encoded proprio

        Returns:
            actions: (B, chunk_size, action_dim)
        """
        B = current_patches.shape[0]

        # Subsample patches if too many
        if current_patches.shape[1] > self.max_patches:
            indices = torch.linspace(0, current_patches.shape[1] - 1, self.max_patches).long()
            current_patches = current_patches[:, indices]
        if goal_patches.shape[1] > self.max_patches:
            indices = torch.linspace(0, goal_patches.shape[1] - 1, self.max_patches).long()
            goal_patches = goal_patches[:, indices]

        # Project patches
        current_tokens = self.patch_proj(current_patches) + self.current_token
        goal_tokens = self.patch_proj(goal_patches) + self.goal_token

        # Project proprio
        proprio_tokens = self.proprio_proj(proprio_emb).unsqueeze(1) + self.proprio_token

        # Concatenate context
        context = torch.cat([current_tokens, goal_tokens, proprio_tokens], dim=1)

        # Action queries
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_encoding(queries)

        # Transformer
        action_features = self.transformer(tgt=queries, memory=context)

        # Output
        actions = self.action_head(action_features)

        return actions


def test_policy_head():
    """Test policy head"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test improved PolicyHead
    print("Testing PolicyHead (multi-token context)...")
    policy = PolicyHead(
        vision_dim=1408,  # Correct ViT-Giant dim
        proprio_dim=256,
        hidden_dim=512,
        action_dim=7,
        chunk_size=50,
        n_context_tokens=4,
    ).to(device)

    # Test inputs
    B = 4
    current_emb = torch.rand(B, 1408).to(device)
    goal_emb = torch.rand(B, 1408).to(device)
    proprio_emb = torch.rand(B, 256).to(device)

    # Forward pass
    actions = policy(current_emb, goal_emb, proprio_emb)

    print(f"  Current emb shape: {current_emb.shape}")
    print(f"  Goal emb shape: {goal_emb.shape}")
    print(f"  Proprio emb shape: {proprio_emb.shape}")
    print(f"  Actions shape: {actions.shape}")  # Should be (4, 50, 7)

    # Check action bounds
    assert actions.min() >= -1.0 and actions.max() <= 1.0, "Actions out of bounds!"
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")

    # Test PolicyHeadWithPatches
    print("\nTesting PolicyHeadWithPatches...")
    policy_patches = PolicyHeadWithPatches(
        vision_dim=1408,
        proprio_dim=256,
        hidden_dim=512,
        action_dim=7,
        chunk_size=50,
        max_patches=256,
    ).to(device)

    # Patch inputs
    current_patches = torch.rand(B, 2048, 1408).to(device)  # Full patch tokens
    goal_patches = torch.rand(B, 256, 1408).to(device)

    actions_patches = policy_patches(current_patches, goal_patches, proprio_emb)
    print(f"  Current patches shape: {current_patches.shape}")
    print(f"  Goal patches shape: {goal_patches.shape}")
    print(f"  Actions shape: {actions_patches.shape}")

    n_params = sum(p.numel() for p in policy_patches.parameters())
    print(f"  Parameters: {n_params / 1e6:.2f}M")

    print("\n✅ Policy head tests passed!")


if __name__ == "__main__":
    test_policy_head()
