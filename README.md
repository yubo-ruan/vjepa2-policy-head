# V-JEPA 2 Policy Head for LIBERO

A goal-conditioned robot manipulation policy using V-JEPA 2 as the visual encoder, designed for the LIBERO benchmark.

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     V-JEPA 2 Policy                         │
                    └─────────────────────────────────────────────────────────────┘
                                              │
           ┌──────────────────────────────────┼──────────────────────────────────┐
           │                                  │                                  │
           ▼                                  ▼                                  ▼
    ┌─────────────┐                   ┌─────────────┐                   ┌─────────────┐
    │ Video Input │                   │ Goal Image  │                   │ Proprio     │
    │ (16 frames) │                   │ (1 frame)   │                   │ (5 history) │
    │ 256×256×3   │                   │ 256×256×3   │                   │ 15-dim      │
    └──────┬──────┘                   └──────┬──────┘                   └──────┬──────┘
           │                                  │                                  │
           ▼                                  ▼                                  │
    ┌─────────────┐                   ┌─────────────┐                           │
    │  V-JEPA 2   │                   │  V-JEPA 2   │                           │
    │ ViT-Giant   │                   │ ViT-Giant   │                           │
    │  (frozen)   │                   │  (frozen)   │                           │
    └──────┬──────┘                   └──────┬──────┘                           │
           │                                  │                                  │
           ▼                                  ▼                                  ▼
    ┌─────────────┐                   ┌─────────────┐                   ┌─────────────┐
    │   Spatial   │                   │   Spatial   │                   │   Proprio   │
    │   Tokens    │                   │   Tokens    │                   │   Encoder   │
    │  (64×1408)  │                   │  (64×1408)  │                   │  MLP→256-D  │
    └──────┬──────┘                   └──────┬──────┘                   └──────┬──────┘
           │                                  │                                  │
           └──────────────────────────────────┼──────────────────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   Policy Head   │
                                    │  (Transformer   │
                                    │   Decoder)      │
                                    │  132 context    │
                                    │  tokens total   │
                                    └────────┬────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │  Action Chunk   │
                                    │   (50 × 7-D)    │
                                    └─────────────────┘
```

## Model Components

### 1. V-JEPA 2 Encoder (Frozen)
- **Model**: ViT-Giant (vjepa2_vitg)
- **Embedding dimension**: 1408
- **Image size**: 256×256
- **Patch size**: 16×16 → 16×16 spatial grid
- **Spatial tokens**: Downsampled to 8×8 = 64 tokens per modality

### 2. Spatial Token Architecture
- 64 spatial tokens from video (8×8 grid)
- 64 spatial tokens from goal image (8×8 grid)
- 4 proprioception tokens
- **Total context**: 64 + 64 + 4 = 132 tokens

### 3. Proprioception Encoder
- **Input**: 15-dimensional state vector
  - End-effector position: 3D
  - End-effector orientation (euler): 3D
  - Gripper state: 2D
  - Joint positions: 7D
- **History**: 5 timesteps
- **Architecture**: MLP (15×5 → 128 → 256)

### 4. Policy Head (Transformer Decoder)
- **Hidden dimension**: 512
- **Attention heads**: 8
- **Layers**: 4
- **Action queries**: 50 (chunk size)
- **Output**: 50 × 7-dimensional actions

## Quick Start

### 1. Precompute Embeddings
```bash
# Basic precomputation
python scripts/precompute.py --suite libero_spatial --output_dir /workspace/data/embeddings

# With static frame augmentation (recommended)
python scripts/precompute.py --suite libero_spatial --output_dir /workspace/data/embeddings --static_aug
```

### 2. Train Policy
```bash
# Basic training
python scripts/train.py --config configs/config.yaml

# With custom options
python scripts/train.py --config configs/config.yaml \
    --epochs 100 \
    --lr 0.0001 \
    --gripper_oversample 5 \
    --gripper_jitter 5
```

### 3. Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --suite libero_spatial \
    --n_episodes 20
```

## Configuration

All settings are in `configs/config.yaml`:

### Model
| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_dim` | 1408 | V-JEPA 2 embedding dimension |
| `hidden_dim` | 512 | Policy head hidden dimension |
| `num_layers` | 4 | Transformer decoder layers |
| `num_heads` | 8 | Attention heads |
| `num_spatial_tokens` | 64 | Tokens per modality (8×8) |
| `chunk_size` | 50 | Action chunk size |
| `action_dim` | 7 | Action dimensions (OSC) |

### Training
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32 | Training batch size |
| `lr` | 1e-4 | Learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `epochs` | 100 | Training epochs |
| `warmup_epochs` | 5 | Linear warmup |
| `grad_clip` | 1.0 | Gradient clipping |

### Data Augmentation
| Parameter | Value | Description |
|-----------|-------|-------------|
| `noise_std` | 0.05 | Gaussian noise on embeddings |
| `normalize` | true | L2 normalize embeddings |
| `gripper_oversample` | 5 | Oversample gripper transitions |
| `gripper_jitter` | 5 | Temporal jitter for transitions |

### Loss Weighting
| Parameter | Value | Description |
|-----------|-------|-------------|
| `gripper_loss_weight` | 2.0 | Extra weight for gripper dim |
| `start_weight` | 3.0 | Weight for first N timesteps |
| `start_steps` | 10 | Number of start timesteps |
| `transition_weight` | 5.0 | Weight for gripper transitions |

## File Structure

```
vjepa2-policy-head/
├── configs/
│   └── config.yaml           # Unified configuration
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── precompute.py         # Embedding precomputation
│   └── debug/                # Debug and diagnostic tools
│       ├── debug_gt_actions.py
│       ├── visualize_policy.py
│       └── ...
├── vjepa_policy/
│   ├── __init__.py           # Package exports
│   ├── models/
│   │   ├── policy.py         # PolicyHead, VJEPA2Policy
│   │   ├── vjepa2_encoder.py # V-JEPA 2 wrapper
│   │   └── full_model.py     # Legacy compatibility
│   ├── data/
│   │   ├── dataset.py        # PolicyDataset
│   │   └── libero_dataset.py # Raw data loading
│   ├── training/
│   │   ├── loss.py           # ActionLoss
│   │   └── trainer.py        # Training loop
│   └── utils/
│       └── evaluation.py     # LIBERO evaluation
└── README.md
```

## LIBERO Benchmark Suites

| Suite | Tasks | Description |
|-------|-------|-------------|
| `libero_spatial` | 10 | Spatial reasoning tasks |
| `libero_object` | 10 | Object manipulation tasks |
| `libero_goal` | 10 | Goal-conditioned tasks |
| `libero_90` | 90 | Large-scale benchmark |
| `libero_10` | 10 | Subset of LIBERO-100 |

## Key Design Decisions

1. **Spatial Tokens Only**: Uses 64 spatial tokens (8×8 grid) per modality instead of mean-pooled embeddings, preserving spatial information.

2. **Frozen V-JEPA 2**: Encoder is frozen to leverage pre-trained representations and reduce training compute.

3. **Action Chunking**: Predicting 50 actions at once reduces compounding errors and improves temporal consistency.

4. **Goal Conditioning**: Using final frame from demonstrations as goal image enables task specification without language.

5. **Gripper-Focused Augmentation**: Oversampling transitions and temporal jitter help learn precise gripper timing.

6. **Weighted Loss**: Higher weights on start timesteps and gripper transitions focus learning on critical moments.

## Action Space

LIBERO uses Operational Space Control (OSC) with 7-dimensional actions:
- Position delta (dx, dy, dz): 3D, range [-1, 1]
- Rotation delta (axis-angle): 3D, range [-1, 1]
- Gripper command: 1D, {-1: close, 1: open}

## Parameter Counts

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| V-JEPA 2 Encoder | ~1.1B | No (frozen) |
| Proprio Encoder | ~45K | Yes |
| Policy Head | ~8.5M | Yes |
| **Total Trainable** | **~8.5M** | - |

## Known Issues

1. **Success Detection**: LIBERO environments don't populate `info['success']`. The evaluation code uses `env.env._check_success()` as a workaround.

2. **Generalization Gap**: The policy may achieve low validation loss but still fail on evaluation due to distribution shift between training trajectories and novel initial states.

## License

This project uses V-JEPA 2 weights which are subject to Meta's license terms.
