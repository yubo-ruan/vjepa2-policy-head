# src/models/__init__.py
# Note: vjepa2_encoder must be imported directly to avoid circular imports
# from .vjepa2_encoder import VJEPA2Encoder

from .proprio_encoder import ProprioEncoder
from .policy_head import PolicyHead
from .full_model import VJEPA2Policy

__all__ = [
    'ProprioEncoder',
    'PolicyHead',
    'VJEPA2Policy',
]
