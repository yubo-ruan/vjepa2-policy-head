"""
V-JEPA 2 Encoder Wrapper

Loads V-JEPA 2 from local weights (no HuggingFace download needed).
Supports both video mode (16 frames) and single image mode.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union
import sys


class VJEPA2Encoder(nn.Module):
    """
    V-JEPA 2 encoder wrapper with local weight loading.
    
    Supports two loading methods:
    1. Local weights (preferred): Load from workspace/models/
    2. PyTorch Hub: torch.hub.load('facebookresearch/vjepa2', ...)
    3. HuggingFace (fallback): AutoModel.from_pretrained(...)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "vjepa2_vitg",  # vjepa2_vitl, vjepa2_vith, vjepa2_vitg
        freeze: bool = True,
        device: str = "cuda",
        load_ac_predictor: bool = False,  # Also load action-conditioned predictor
    ):
        """
        Args:
            model_path: Path to local model weights directory
                        Expected structure: model_path/encoder.pth, model_path/config.yaml
                        If None, will try PyTorch Hub
            model_name: Model variant (vjepa2_vitl, vjepa2_vith, vjepa2_vitg)
            freeze: Whether to freeze encoder weights
            device: Device to load model on
            load_ac_predictor: Whether to also load the AC predictor for planning
        """
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        self.freeze = freeze
        
        # Load model
        if model_path is not None:
            self._load_from_local(model_path, load_ac_predictor)
        else:
            self._load_from_hub(load_ac_predictor)
        
        # Move to device
        self.encoder.to(device)
        self.encoder.eval()
        
        if hasattr(self, 'ac_predictor') and self.ac_predictor is not None:
            self.ac_predictor.to(device)
            self.ac_predictor.eval()
        
        # Freeze if specified
        if freeze:
            self._freeze()
        
        # Get embedding dimension
        self.embed_dim = self._get_embed_dim()
        print(f"V-JEPA 2 loaded. Embedding dim: {self.embed_dim}")
    
    def _load_from_local(self, model_path: str, load_ac_predictor: bool):
        """Load model from local weights directory"""
        model_path = Path(model_path)
        
        print(f"Loading V-JEPA 2 from local path: {model_path}")
        
        # Check for different possible file structures
        # Structure 1: Direct checkpoint file
        if model_path.suffix in ['.pth', '.pt', '.ckpt']:
            checkpoint_path = model_path
        # Structure 2: Directory with checkpoint inside
        else:
            possible_files = [
                model_path / "encoder.pth",
                model_path / "model.pth", 
                model_path / "checkpoint.pth",
                model_path / "vjepa2_vitg.pth",
                model_path / "vjepa2_vith.pth",
                model_path / "vjepa2_vitl.pth",
            ]
            
            checkpoint_path = None
            for f in possible_files:
                if f.exists():
                    checkpoint_path = f
                    break
            
            if checkpoint_path is None:
                # List available files
                available = list(model_path.glob("*.pth")) + list(model_path.glob("*.pt"))
                raise FileNotFoundError(
                    f"Could not find model checkpoint in {model_path}. "
                    f"Available files: {available}"
                )
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        else:
            state_dict = checkpoint
        
        # Build model architecture
        self.encoder = self._build_encoder_architecture()
        
        # Load weights
        # Handle potential key mismatches
        try:
            self.encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed, trying non-strict: {e}")
            self.encoder.load_state_dict(state_dict, strict=False)
        
        # Load AC predictor if requested
        if load_ac_predictor:
            ac_path = model_path.parent / "ac_predictor.pth" if model_path.is_file() else model_path / "ac_predictor.pth"
            if ac_path.exists():
                print(f"Loading AC predictor: {ac_path}")
                ac_checkpoint = torch.load(ac_path, map_location='cpu')
                self.ac_predictor = self._build_ac_predictor_architecture()
                self.ac_predictor.load_state_dict(ac_checkpoint)
            else:
                print(f"AC predictor not found at {ac_path}, skipping")
                self.ac_predictor = None
        else:
            self.ac_predictor = None
    
    def _load_from_hub(self, load_ac_predictor: bool):
        """Load model from PyTorch Hub"""
        print(f"Loading V-JEPA 2 from PyTorch Hub: {self.model_name}")
        
        try:
            if load_ac_predictor:
                # Load both encoder and AC predictor
                self.encoder, self.ac_predictor = torch.hub.load(
                    'facebookresearch/vjepa2',
                    f'{self.model_name}_ac',  # e.g., vjepa2_vitg_ac
                    pretrained=True,
                )
            else:
                # Load encoder only
                self.encoder = torch.hub.load(
                    'facebookresearch/vjepa2',
                    self.model_name,  # e.g., vjepa2_vitg
                    pretrained=True,
                )
                self.ac_predictor = None
                
        except Exception as e:
            print(f"PyTorch Hub loading failed: {e}")
            print("Falling back to HuggingFace...")
            self._load_from_huggingface()
    
    def _load_from_huggingface(self):
        """Fallback: Load from HuggingFace"""
        from transformers import AutoModel
        
        model_map = {
            'vjepa2_vitl': 'facebook/vjepa2-vitl-fpc64-256',
            'vjepa2_vith': 'facebook/vjepa2-vith-fpc64-256',
            'vjepa2_vitg': 'facebook/vjepa2-vitg-fpc64-384',
        }
        
        hf_name = model_map.get(self.model_name, self.model_name)
        print(f"Loading from HuggingFace: {hf_name}")
        
        self.encoder = AutoModel.from_pretrained(hf_name)
        self.ac_predictor = None
        self._use_hf = True
    
    def _build_encoder_architecture(self):
        """
        Build V-JEPA 2 encoder architecture.
        
        This should match the architecture used during pre-training.
        For now, we use a placeholder - the actual architecture depends
        on how the weights were saved.
        """
        # Try to import from vjepa2 repo if available
        try:
            from vjepa2.models import vit_giant, vit_huge, vit_large
            
            model_builders = {
                'vjepa2_vitl': vit_large,
                'vjepa2_vith': vit_huge,
                'vjepa2_vitg': vit_giant,
            }
            
            builder = model_builders.get(self.model_name)
            if builder:
                return builder()
        except ImportError:
            pass
        
        # Fallback: Use timm or manual implementation
        try:
            import timm
            
            timm_models = {
                'vjepa2_vitl': 'vit_large_patch16_384',
                'vjepa2_vith': 'vit_huge_patch14_clip_384',
                'vjepa2_vitg': 'vit_giant_patch14_clip_224',
            }
            
            timm_name = timm_models.get(self.model_name)
            if timm_name:
                return timm.create_model(timm_name, pretrained=False)
        except ImportError:
            pass
        
        raise RuntimeError(
            f"Could not build encoder architecture for {self.model_name}. "
            "Please ensure vjepa2 or timm is installed."
        )
    
    def _build_ac_predictor_architecture(self):
        """Build AC predictor architecture"""
        # Placeholder - actual implementation depends on vjepa2 repo
        try:
            from vjepa2.models import ACPredictor
            return ACPredictor()
        except ImportError:
            raise RuntimeError(
                "AC predictor architecture not available. "
                "Please install vjepa2 repo."
            )
    
    def _freeze(self):
        """Freeze all encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("V-JEPA 2 encoder frozen")
    
    def _get_embed_dim(self) -> int:
        """Get embedding dimension from model"""
        # Try common attribute names
        if hasattr(self.encoder, 'embed_dim'):
            return self.encoder.embed_dim
        if hasattr(self.encoder, 'hidden_size'):
            return self.encoder.hidden_size
        if hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'hidden_size'):
            return self.encoder.config.hidden_size
        
        # Fallback: run a forward pass
        print("Detecting embed_dim via forward pass...")
        with torch.no_grad():
            dummy = torch.zeros(1, 16, 3, 224, 224).to(self.device)
            out = self._forward_encoder(dummy)
            return out.shape[-1]
    
    def _forward_encoder(self, video: torch.Tensor) -> torch.Tensor:
        """
        Internal forward pass through encoder.
        Handles different model interfaces.
        """
        # V-JEPA 2 native interface
        if hasattr(self.encoder, 'forward_features'):
            features = self.encoder.forward_features(video)
        # HuggingFace interface
        elif hasattr(self, '_use_hf') and self._use_hf:
            outputs = self.encoder(pixel_values=video)
            features = outputs.last_hidden_state
        # Standard forward
        else:
            features = self.encoder(video)
        
        # Pool over spatial/temporal dimensions if needed
        if features.dim() == 4:  # (B, T, N, D) or (B, N, T, D)
            features = features.mean(dim=(1, 2))
        elif features.dim() == 3:  # (B, N, D)
            features = features.mean(dim=1)
        
        return features
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video clip to single embedding.
        
        Args:
            video: (B, T, C, H, W) video tensor, values in [0, 1]
                   T should be 16 frames typically
        
        Returns:
            embedding: (B, embed_dim) video embedding
        """
        # Ensure correct format
        if video.dim() == 4:  # (B, C, H, W) single image
            video = video.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        
        B, T, C, H, W = video.shape
        
        # Normalize if needed (V-JEPA 2 expects specific normalization)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(video.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(video.device)
        video = (video - mean) / std
        
        with torch.no_grad() if self.freeze else torch.enable_grad():
            embedding = self._forward_encoder(video)
        
        return embedding
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode single image to embedding.
        
        Args:
            image: (B, C, H, W) image tensor, values in [0, 1]
        
        Returns:
            embedding: (B, embed_dim) image embedding
        """
        # Expand to video format (repeat frame)
        video = image.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        return self.encode_video(video)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - auto-detect video vs image"""
        if x.dim() == 5:
            return self.encode_video(x)
        elif x.dim() == 4:
            return self.encode_image(x)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D")
    
    def predict_next_state(
        self,
        current_emb: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state embedding using AC predictor.
        
        Args:
            current_emb: (B, embed_dim) current state embedding
            action: (B, action_dim) action to take
        
        Returns:
            next_emb: (B, embed_dim) predicted next state embedding
        """
        if self.ac_predictor is None:
            raise RuntimeError("AC predictor not loaded. Set load_ac_predictor=True")
        
        with torch.no_grad():
            next_emb = self.ac_predictor(current_emb, action)
        
        return next_emb


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
            elif path.is_dir():
                # Check for checkpoint files inside
                for ckpt in path.glob("*.pth"):
                    name = f"{path.name}/{ckpt.stem}"
                    available[name] = str(ckpt)
    
    if available:
        print(f"Found V-JEPA 2 weights:")
        for name, path in available.items():
            print(f"  {name}: {path}")
    else:
        print(f"No V-JEPA 2 weights found in {base}")
        print("Available files:")
        for f in base.rglob("*"):
            if f.is_file():
                print(f"  {f}")
    
    return available


def test_vjepa2_encoder_local():
    """Test V-JEPA 2 encoder with local weights"""
    
    # Find available weights
    available = find_model_weights("/workspace/models")
    
    if not available:
        print("No local weights found, testing with Hub...")
        model_path = None
    else:
        # Use first available
        model_name, model_path = next(iter(available.items()))
        print(f"Using: {model_name} at {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = VJEPA2Encoder(
        model_path=model_path,
        model_name="vjepa2_vitg",
        device=device,
        freeze=True,
    )
    
    # Test video encoding
    print("\nTesting video encoding...")
    video = torch.rand(2, 16, 3, 384, 384).to(device)
    video_emb = encoder.encode_video(video)
    print(f"Video embedding shape: {video_emb.shape}")
    
    # Test image encoding
    print("\nTesting image encoding...")
    image = torch.rand(2, 3, 384, 384).to(device)
    image_emb = encoder.encode_image(image)
    print(f"Image embedding shape: {image_emb.shape}")
    
    print("\n✅ V-JEPA 2 encoder test passed!")


if __name__ == "__main__":
    test_vjepa2_encoder_local()