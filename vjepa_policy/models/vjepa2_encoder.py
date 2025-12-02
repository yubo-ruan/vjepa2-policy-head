"""
V-JEPA 2 Encoder Wrapper

Self-contained implementation using local vjepa_src copy.
Loads from local checkpoint (encoder + AC predictor).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Import from local vjepa_src (self-contained, no external dependency)
from vjepa_src.models import vision_transformer as vjepa_vit
from vjepa_src.models import ac_predictor as vjepa_ac
from vjepa_src.models import attentive_pooler as vjepa_pooler
from vjepa_src.hub import backbones as vjepa_backbones


def _get_vjepa_modules():
    """Return pre-imported vjepa modules"""
    return vjepa_backbones, vjepa_vit, vjepa_ac, vjepa_pooler


class VJEPA2Encoder(nn.Module):
    """
    V-JEPA 2 encoder wrapper using official Meta implementation.

    Key details:
    - ViT-Giant embed_dim = 1408
    - Input format: (B, C, T, H, W) - channels first, then time
    - Uses RoPE (Rotary Position Embeddings)
    - Output: patch tokens (B, N, D) where N = (T/2) * (H/16) * (W/16)
    """

    # Model configurations
    MODEL_CONFIGS = {
        'vjepa2_vitl': {'img_size': 256, 'embed_dim': 1024},
        'vjepa2_vith': {'img_size': 256, 'embed_dim': 1280},
        'vjepa2_vitg': {'img_size': 256, 'embed_dim': 1408},
        'vjepa2_vitg_384': {'img_size': 384, 'embed_dim': 1408},
    }

    def __init__(
        self,
        model_path: Optional[str] = "/workspace/models/vjepa2-ac-vitg.pt",
        model_name: str = "vjepa2_vitg",
        freeze: bool = True,
        device: str = "cuda",
        num_frames: int = 16,
        use_attentive_pool: bool = True,
        pool_num_queries: int = 1,
        load_ac_predictor: bool = False,
    ):
        """
        Args:
            model_path: Path to checkpoint file (e.g., vjepa2-ac-vitg.pt)
            model_name: Model variant (vjepa2_vitl, vjepa2_vith, vjepa2_vitg)
            freeze: Whether to freeze encoder weights
            device: Device to load model on
            num_frames: Number of input frames (default 16)
            use_attentive_pool: Use AttentivePooler for aggregation (recommended)
            pool_num_queries: Number of query tokens for AttentivePooler
            load_ac_predictor: Whether to load AC predictor (saves memory if False)
        """
        super().__init__()
        self.load_ac_predictor = load_ac_predictor

        self.device = device
        self.model_name = model_name
        self.freeze_encoder = freeze
        self.num_frames = num_frames
        self.use_attentive_pool = use_attentive_pool

        # Get model config
        config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS['vjepa2_vitg'])
        self.img_size = config['img_size']
        self.embed_dim = config['embed_dim']

        # Load V-JEPA modules
        self._vjepa_modules = _get_vjepa_modules()

        # Load model
        self._load_model(model_path)

        # Create pooler if requested
        if use_attentive_pool:
            self._create_attentive_pooler(pool_num_queries)

        # Move to device
        self.encoder.to(device)
        if hasattr(self, 'ac_predictor') and self.ac_predictor is not None:
            self.ac_predictor.to(device)
        if hasattr(self, 'pooler'):
            self.pooler.to(device)

        # Freeze encoder if specified
        if freeze:
            self._freeze_encoder()

        print(f"V-JEPA 2 loaded: {model_name}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Image size: {self.img_size}")
        print(f"  Num frames: {self.num_frames}")
        print(f"  Pooling: {'AttentivePooler' if use_attentive_pool else 'Mean'}")
        print(f"  Frozen: {freeze}")

    def _load_model(self, model_path: Optional[str]):
        """Load encoder and optionally AC predictor from checkpoint"""

        if model_path and Path(model_path).exists():
            self._load_from_local(model_path)
        else:
            self._load_from_hub()

    def _load_from_local(self, model_path: str):
        """Load from local checkpoint file"""
        vjepa_backbones, vit_encoder, vit_ac_predictor, _ = self._vjepa_modules

        def _clean_backbone_key(state_dict):
            """Clean backbone keys from state dict"""
            for key, val in state_dict.copy().items():
                _ = state_dict.pop(key)
                key = key.replace("module.", "")
                key = key.replace("backbone.", "")
                state_dict[key] = val
            return state_dict

        print(f"Loading V-JEPA 2 from local: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Determine model variant from checkpoint or model_name
        img_size = self.img_size

        # Build encoder architecture
        vit_encoder_kwargs = dict(
            patch_size=16,
            img_size=(img_size, img_size),
            num_frames=self.num_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )

        # Use ViT-Giant architecture
        self.encoder = vit_encoder.vit_giant_xformers(**vit_encoder_kwargs)

        # Load encoder weights
        if 'encoder' in checkpoint:
            encoder_state_dict = _clean_backbone_key(checkpoint['encoder'])
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print("  Loaded encoder weights")

        # Build and load AC predictor if available and requested
        if 'predictor' in checkpoint and self.load_ac_predictor:
            vit_predictor_kwargs = dict(
                img_size=(img_size, img_size),
                patch_size=16,
                num_frames=self.num_frames,
                tubelet_size=2,
                embed_dim=self.embed_dim,
            )
            self.ac_predictor = vit_ac_predictor.vit_ac_predictor(**vit_predictor_kwargs)
            predictor_state_dict = _clean_backbone_key(checkpoint['predictor'])
            self.ac_predictor.load_state_dict(predictor_state_dict, strict=True)
            print("  Loaded AC predictor weights")
        else:
            self.ac_predictor = None
            if 'predictor' in checkpoint and not self.load_ac_predictor:
                print("  Skipped AC predictor loading (not needed, saves memory)")

    def _load_from_hub(self):
        """Load from PyTorch Hub as fallback"""
        vjepa_backbones, _, _, _ = self._vjepa_modules

        print(f"Loading V-JEPA 2 from PyTorch Hub...")

        try:
            # Try to load AC model (encoder + predictor)
            self.encoder, self.ac_predictor = vjepa_backbones.vjepa2_ac_vit_giant(
                pretrained=True,
                num_frames=self.num_frames,
            )
        except Exception as e:
            print(f"AC model loading failed: {e}")
            # Fall back to encoder only
            self.encoder, _ = vjepa_backbones.vjepa2_vit_giant(
                pretrained=True,
                num_frames=self.num_frames,
            )
            self.ac_predictor = None

    def _create_attentive_pooler(self, num_queries: int = 1):
        """Create AttentivePooler for aggregating patch tokens"""
        _, _, _, att_pooler = self._vjepa_modules
        AttentivePooler = att_pooler.AttentivePooler

        self.pooler = AttentivePooler(
            num_queries=num_queries,
            embed_dim=self.embed_dim,
            num_heads=16,
            mlp_ratio=4.0,
            depth=1,
        )
        # Pooler is trainable by default
        print(f"  Created AttentivePooler with {num_queries} queries")

    def _freeze_encoder(self):
        """Freeze encoder parameters (pooler remains trainable)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.ac_predictor is not None:
            for param in self.ac_predictor.parameters():
                param.requires_grad = False
        print("  Encoder frozen")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization"""
        # x should be in [0, 1] range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)

        # Handle both (B, C, T, H, W) and (B, C, H, W)
        if x.dim() == 5:
            mean = mean.view(1, 3, 1, 1, 1)
            std = std.view(1, 3, 1, 1, 1)
        else:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)

        return (x - mean) / std

    def encode_patches(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to patch tokens (no pooling).

        Args:
            video: (B, C, T, H, W) video tensor, values in [0, 1]
                   Note: channels first, then time!

        Returns:
            patches: (B, N, D) patch tokens
                     N = (T/tubelet_size) * (H/patch_size) * (W/patch_size)
        """
        # Normalize
        video = self._normalize(video)

        # Forward through encoder
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            patches = self.encoder(video)

        return patches

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video clip to embedding(s).

        Args:
            video: (B, C, T, H, W) video tensor, values in [0, 1]
                   OR (B, T, C, H, W) - will be permuted automatically

        Returns:
            embedding: (B, embed_dim) if pool_num_queries=1
                      (B, num_queries, embed_dim) if pool_num_queries>1
        """
        # Handle both input formats
        if video.dim() == 5:
            B, dim2, dim3, H, W = video.shape
            # Check if input is (B, T, C, H, W) and needs permutation
            if dim3 == 3 and dim2 != 3:  # dim3 is C=3, dim2 is T
                video = video.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)

        # Get patch tokens
        patches = self.encode_patches(video)

        # Pool to get embedding
        if self.use_attentive_pool:
            embedding = self.pooler(patches)
            if embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)  # (B, D)
        else:
            embedding = patches.mean(dim=1)  # (B, D)

        return embedding

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode single image to embedding.

        Args:
            image: (B, C, H, W) image tensor, values in [0, 1]

        Returns:
            embedding: (B, embed_dim) image embedding
        """
        # Expand to video format: (B, C, H, W) -> (B, C, 2, H, W)
        # Use 2 frames (1 tubelet) for minimal computation
        video = image.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        return self.encode_video(video)

    def encode_video_spatial(self, video: torch.Tensor, target_spatial: int = 8) -> torch.Tensor:
        """
        Encode video to spatial tokens (preserving spatial structure).

        Instead of pooling all tokens to 1, downsample spatially to preserve
        spatial information. This helps with static frame generalization.

        Args:
            video: (B, C, T, H, W) video tensor, values in [0, 1]
                   OR (B, T, C, H, W) - will be permuted automatically
            target_spatial: Target spatial resolution (8 -> 64 tokens)

        Returns:
            spatial_tokens: (B, target_spatial^2, embed_dim) = (B, 64, 1408)
        """
        # Handle both input formats
        if video.dim() == 5:
            B, dim2, dim3, H, W = video.shape
            if dim3 == 3 and dim2 != 3:  # (B, T, C, H, W)
                video = video.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)

        # Get patch tokens: (B, N, D) where N = T/2 * H/16 * W/16
        patches = self.encode_patches(video)
        B, N, D = patches.shape

        # Compute spatial dimensions
        # For 16 frames, 256x256: N = 8 * 16 * 16 = 2048 (temporal * spatial)
        # For 16 frames: temporal_size = 16/2 = 8
        # spatial_size = sqrt(N / temporal_size) = sqrt(2048/8) = 16
        temporal_size = self.num_frames // 2  # tubelet_size = 2
        spatial_total = N // temporal_size
        spatial_size = int(spatial_total ** 0.5)  # Should be 16 for 256x256

        # Reshape to (B, T, H_spatial, W_spatial, D)
        patches = patches.view(B, temporal_size, spatial_size, spatial_size, D)

        # Average over temporal dimension: (B, H_spatial, W_spatial, D)
        patches = patches.mean(dim=1)

        # Rearrange for spatial pooling: (B, D, H, W)
        patches = patches.permute(0, 3, 1, 2)

        # Spatial downsampling: (B, D, 8, 8)
        pool = nn.AdaptiveAvgPool2d((target_spatial, target_spatial))
        patches = pool(patches)

        # Reshape to (B, 64, D)
        patches = patches.permute(0, 2, 3, 1).reshape(B, target_spatial * target_spatial, D)

        return patches

    def encode_image_spatial(self, image: torch.Tensor, target_spatial: int = 8) -> torch.Tensor:
        """
        Encode single image to spatial tokens.

        For images, we only have 1 tubelet (2 frames), so we handle the
        reshape differently than for video.

        Args:
            image: (B, C, H, W) image tensor, values in [0, 1]
            target_spatial: Target spatial resolution (8 -> 64 tokens)

        Returns:
            spatial_tokens: (B, target_spatial^2, embed_dim) = (B, 64, 1408)
        """
        # Expand to video format: (B, C, H, W) -> (B, C, 2, H, W)
        video = image.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        # Get patch tokens: (B, N, D) where N = 1 * 16 * 16 = 256 (for 2 frames)
        patches = self.encode_patches(video)
        B, N, D = patches.shape

        # For image: N = 1 * spatial_size * spatial_size = 256
        # So spatial_size = sqrt(256) = 16
        spatial_size = int(N ** 0.5)

        # Reshape to (B, H_spatial, W_spatial, D)
        patches = patches.view(B, spatial_size, spatial_size, D)

        # Rearrange for spatial pooling: (B, D, H, W)
        patches = patches.permute(0, 3, 1, 2)

        # Spatial downsampling: (B, D, 8, 8)
        pool = nn.AdaptiveAvgPool2d((target_spatial, target_spatial))
        patches = pool(patches)

        # Reshape to (B, 64, D)
        patches = patches.permute(0, 2, 3, 1).reshape(B, target_spatial * target_spatial, D)

        return patches

    def forward(
        self,
        x: torch.Tensor,
        return_patches: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass - auto-detect video vs image.

        Args:
            x: Input tensor
               - 5D (B, C, T, H, W) or (B, T, C, H, W): video
               - 4D (B, C, H, W): image
            return_patches: If True, return patch tokens instead of pooled embedding

        Returns:
            embedding or patches depending on return_patches
        """
        if x.dim() == 5:
            if return_patches:
                # Handle format conversion
                B, dim2, dim3, H, W = x.shape
                if dim3 == 3 and dim2 != 3:
                    x = x.permute(0, 2, 1, 3, 4)
                return self.encode_patches(x)
            return self.encode_video(x)
        elif x.dim() == 4:
            if return_patches:
                video = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)
                return self.encode_patches(video)
            return self.encode_image(x)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D")

    def get_trainable_params(self):
        """Get trainable parameters (pooler if encoder is frozen)"""
        params = []
        if self.use_attentive_pool:
            params.extend(self.pooler.parameters())
        if not self.freeze_encoder:
            params.extend(self.encoder.parameters())
        return params


def find_model_weights(base_path: str = "/workspace/models") -> dict:
    """
    Find available V-JEPA 2 weights in the workspace.

    Returns dict of {model_name: path}
    """
    base = Path(base_path)

    if not base.exists():
        print(f"Models directory not found: {base}")
        return {}

    available = {}

    # Search for V-JEPA 2 related files
    patterns = ["*vjepa*", "*VJEPA*", "*jepa*"]

    for pattern in patterns:
        for path in base.rglob(pattern):
            if path.is_file() and path.suffix in ['.pth', '.pt', '.ckpt']:
                name = path.stem
                available[name] = str(path)

    if available:
        print(f"Found V-JEPA 2 weights:")
        for name, path in available.items():
            print(f"  {name}: {path}")
    else:
        print(f"No V-JEPA 2 weights found in {base}")
        for f in base.iterdir():
            print(f"  {f}")

    return available


def test_vjepa2_encoder():
    """Test V-JEPA 2 encoder with local weights"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find weights
    available = find_model_weights("/workspace/models")
    model_path = available.get('vjepa2-ac-vitg', "/workspace/models/vjepa2-ac-vitg.pt")

    # Create encoder
    encoder = VJEPA2Encoder(
        model_path=model_path,
        model_name="vjepa2_vitg",
        device=device,
        freeze=True,
        num_frames=16,
        use_attentive_pool=True,
    )

    # Test video encoding (B, C, T, H, W) format
    print("\nTesting video encoding (B, C, T, H, W)...")
    video = torch.rand(2, 3, 16, 256, 256).to(device)
    video_emb = encoder.encode_video(video)
    print(f"  Input shape: {video.shape}")
    print(f"  Output shape: {video_emb.shape}")
    assert video_emb.shape == (2, 1408), f"Expected (2, 1408), got {video_emb.shape}"

    # Test video encoding (B, T, C, H, W) format - should auto-convert
    print("\nTesting video encoding (B, T, C, H, W) - auto-convert...")
    video_alt = torch.rand(2, 16, 3, 256, 256).to(device)
    video_emb_alt = encoder.encode_video(video_alt)
    print(f"  Input shape: {video_alt.shape}")
    print(f"  Output shape: {video_emb_alt.shape}")
    assert video_emb_alt.shape == (2, 1408)

    # Test image encoding
    print("\nTesting image encoding...")
    image = torch.rand(2, 3, 256, 256).to(device)
    image_emb = encoder.encode_image(image)
    print(f"  Input shape: {image.shape}")
    print(f"  Output shape: {image_emb.shape}")
    assert image_emb.shape == (2, 1408)

    # Test patch tokens
    print("\nTesting patch token extraction...")
    patches = encoder(video, return_patches=True)
    print(f"  Patch tokens shape: {patches.shape}")
    # Expected: (B, N, D) where N = (16/2) * (256/16) * (256/16) = 8 * 16 * 16 = 2048
    expected_n = (16 // 2) * (256 // 16) * (256 // 16)
    assert patches.shape == (2, expected_n, 1408), f"Expected (2, {expected_n}, 1408), got {patches.shape}"

    print("\n✅ V-JEPA 2 encoder test passed!")


if __name__ == "__main__":
    test_vjepa2_encoder()
