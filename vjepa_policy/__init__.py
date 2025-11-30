# V-JEPA 2 Policy Package
#
# Simplified architecture: Spatial tokens only (64 per modality, 8x8 grid)
# No mean-pooling variants.

from vjepa_policy.models.policy import (
    PolicyHead,
    VJEPA2Policy,
    ProprioEncoder,
    create_policy_head,
    create_model,
)
from vjepa_policy.data.dataset import (
    PolicyDataset,
    create_dataloaders,
)
from vjepa_policy.training.loss import (
    ActionLoss,
    create_loss,
)
from vjepa_policy.training.trainer import (
    Trainer,
    set_seed,
)

__all__ = [
    # Models
    'PolicyHead',
    'VJEPA2Policy',
    'ProprioEncoder',
    'create_policy_head',
    'create_model',
    # Data
    'PolicyDataset',
    'create_dataloaders',
    # Loss
    'ActionLoss',
    'create_loss',
    # Training
    'Trainer',
    'set_seed',
]

__version__ = '2.0.0'  # Major version bump for simplified architecture
