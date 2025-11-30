# V-JEPA 2 Policy Models Module
#
# Spatial tokens only architecture (64 tokens per modality, 8x8 grid)

from .policy import (
    PolicyHead,
    VJEPA2Policy,
    ProprioEncoder,
    create_policy_head,
    create_model,
)

# Legacy - kept for backward compatibility
from .full_model import VJEPA2PolicySpatial

__all__ = [
    # Primary - unified policy
    'PolicyHead',
    'VJEPA2Policy',
    'ProprioEncoder',
    'create_policy_head',
    'create_model',
    # Legacy
    'VJEPA2PolicySpatial',
]
