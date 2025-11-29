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
    │  Embedding  │                   │  Embedding  │                   │   Proprio   │
    │   (1408-D)  │                   │   (1408-D)  │                   │   Encoder   │
    │  or (64×1408)│                  │  or (64×1408)│                  │  MLP→256-D  │
    └──────┬──────┘                   └──────┬──────┘                   └──────┬──────┘
           │                                  │                                  │
           └──────────────────────────────────┼──────────────────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   Policy Head   │
                                    │  (Transformer   │
                                    │   Decoder)      │
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
- **Patch size**: 16×16 → 16×16 spatial patches
- **Tubelet size**: 2 (temporal)
- **Video tokens**: 16 frames → 8 temporal × 256 spatial = 2048 tokens
- **Image tokens**: 1 frame (repeated 2×) → 256 tokens

### 2. Pooling Strategies

#### Mean Pooling (Default)
- Pools all tokens to single 1408-D vector
- Total context: 4 video + 4 goal + 4 proprio = 12 tokens

#### Spatial Pooling (Experimental)
- Downsamples 16×16 spatial grid to 8×8 using AdaptiveAvgPool2d
- Preserves 64 spatial tokens per modality
- Total context: 64 video + 64 goal + 4 proprio = 132 tokens

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

## Hyperparameters

### Training Configuration (`configs/default.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `epochs` | 100 | Training epochs |
| `warmup_epochs` | 5 | Linear warmup epochs |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `seed` | 42 | Random seed |

### Data Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_frames` | 16 | Video input frames |
| `proprio_history` | 5 | Proprioception history length |
| `chunk_size` | 50 | Action chunk size |
| `action_dim` | 7 | Action dimensions (OSC) |
| `train_ratio` | 0.9 | Train/val split |
| `sample_stride` | 5 | Reduce sample overlap |

### Robust Embedding Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `noise_std` | 0.05 | Gaussian noise on embeddings |
| `normalize` | true | L2 normalize embeddings |

### Static Frame Augmentation (Spatial Mode)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `static_prob` | 0.25 | Fully static video (first frame repeated) |
| `beginning_prob` | 0.25 | Sample from first 30 frames |

### Evaluation Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `execute_steps` | 10 | Actions before replanning |
| `max_episode_steps` | 300 | Maximum episode length |
| `n_episodes_per_task` | 20 | Episodes per evaluation task |

## Parameter Counts

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| V-JEPA 2 Encoder | ~1.1B | No (frozen) |
| Proprio Encoder | ~45K | Yes |
| Policy Head | ~8.5M | Yes |
| **Total Trainable** | **~8.5M** | - |

## Action Space

LIBERO uses Operational Space Control (OSC) with 7-dimensional actions:
- Position delta (dx, dy, dz): 3D, range [-1, 1]
- Rotation delta (axis-angle): 3D, range [-1, 1]
- Gripper command: 1D, {-1: close, 1: open}

## Training Pipeline

### Option 1: Fast Training (Precomputed Embeddings)
```bash
# Step 1: Precompute V-JEPA 2 embeddings
python scripts/precompute_embeddings.py --suite libero_spatial

# Step 2: Train policy head only
python scripts/train_fast.py --suite libero_spatial --epochs 100
```

### Option 2: Spatial Token Training
```bash
# Step 1: Precompute spatial embeddings (64 tokens)
python scripts/precompute_spatial_embeddings.py --suite libero_spatial

# Step 2: Train with spatial architecture
python scripts/train_spatial.py --suite libero_spatial --config configs/spatial.yaml
```

### Option 3: End-to-End (Slow)
```bash
python scripts/train.py --suite libero_spatial --epochs 100
```

## Evaluation

```bash
# Evaluate mean-pooled model
python scripts/evaluate.py \
    --checkpoint /path/to/best_model.pt \
    --suite libero_spatial \
    --n_episodes 20

# Evaluate spatial model
python scripts/evaluate_spatial.py \
    --checkpoint /path/to/best_model.pt \
    --config configs/spatial.yaml \
    --suite libero_spatial
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

1. **Frozen V-JEPA 2**: The encoder is frozen to leverage pre-trained representations and reduce training compute.

2. **Action Chunking**: Predicting 50 actions at once reduces compounding errors and improves temporal consistency.

3. **Goal Conditioning**: Using final frame from demonstrations as goal image enables task specification without language.

4. **Static Frame Augmentation**: Training with augmented static/beginning-biased videos helps bridge the distribution gap between training (motion) and evaluation (static start).

5. **Embedding Normalization + Noise**: L2 normalization and Gaussian noise during training improves robustness to embedding variations.

## File Structure

```
vjepa2-policy-head/
├── configs/
│   ├── default.yaml          # Mean-pooled configuration
│   └── spatial.yaml          # Spatial token configuration
├── scripts/
│   ├── train.py              # End-to-end training
│   ├── train_fast.py         # Precomputed embedding training
│   ├── train_spatial.py      # Spatial token training
│   ├── precompute_embeddings.py
│   ├── precompute_spatial_embeddings.py
│   ├── evaluate.py           # Mean-pooled evaluation
│   └── evaluate_spatial.py   # Spatial token evaluation
├── vjepa_policy/
│   ├── models/
│   │   ├── vjepa2_encoder.py # V-JEPA 2 wrapper
│   │   ├── policy_head.py    # Transformer decoder
│   │   ├── proprio_encoder.py
│   │   └── full_model.py     # Complete policy
│   ├── data/
│   │   └── libero_dataset.py # Dataset classes
│   ├── training/
│   │   └── trainer.py        # Training loop
│   └── utils/
│       └── evaluation.py     # LIBERO evaluation
└── README.md
```

## Debug Scripts

The `scripts/` directory includes diagnostic tools:

| Script | Description |
|--------|-------------|
| `debug_gt_actions.py` | Replay ground truth actions in environment |
| `debug_init_state.py` | Inspect demo HDF5 structure and initial states |
| `debug_replay_with_state.py` | Replay demos with restored initial state |
| `test_receding_horizon.py` | Compare different `execute_steps` values |
| `test_action_robustness.py` | Compare predictions to GT and test noise tolerance |

## Known Issues

1. **Success Detection**: LIBERO environments don't populate `info['success']`. The evaluation code uses `env.env._check_success()` as a workaround.

2. **Generalization Gap**: The policy may achieve low validation loss but still fail on evaluation due to distribution shift between training trajectories and novel initial states.

## Diagnostic Findings

### Action Noise Tolerance
Ground truth actions tolerate up to ~0.1 noise standard deviation while maintaining 100% success. At 0.2 noise, tasks fail.

### Model Prediction Analysis
- **Position errors**: ~0.27 mean (moderate, some timesteps within tolerance)
- **Rotation errors**: ~0.05 mean (low, within tolerance)
- **Gripper errors**: ~0.29 mean (high variance due to binary action)

The model performs poorly at trajectory start (high position error ~0.5) and during gripper transitions. These critical moments cause task failure even with perfect actions elsewhere.

### Receding Horizon
Testing `execute_steps=1` vs `execute_steps=10` showed no improvement. The issue is not replanning frequency but rather fundamental prediction errors at critical phases.

## Training Results

### Latest Experiment (Spatial + Weighted Loss + Augmentation)

**Configuration:**
- Architecture: Spatial tokens (64 video + 64 goal + 4 proprio = 132 context tokens)
- Data: 50% augmentation (25% static + 25% beginning-biased)
- Loss: Weighted temporal (3x for first 10 timesteps, 5x for gripper transitions)
- Training: 100 epochs, batch size 32, lr=1e-4

**Training Metrics:**
| Metric | Value |
|--------|-------|
| Best Val Loss | 0.1051 (epoch 48) |
| Final Train Loss | 0.035 |
| Final Val Loss | 0.107 |

**Evaluation Results:**
| Suite | Success Rate | Notes |
|-------|--------------|-------|
| libero_spatial (Task 0) | 0% (0/2) | Model struggles with pick-and-place |

### Improvements Implemented

1. **Weighted Temporal Loss** (`vjepa_policy/training/losses.py`)
   - 3x weight on first 10 timesteps (addresses ~0.5 start error)
   - 5x weight on gripper transitions (addresses ~1.99 transition error)

2. **Static Frame Augmentation** (`scripts/precompute_spatial_embeddings.py`)
   - 25% fully static videos (first frame repeated)
   - 25% beginning-biased sampling (first 30 frames)
   - Bridges train/eval distribution gap

3. **Visualization Tools** (`scripts/visualize_policy.py`)
   - GIF generation for VSCode viewing
   - Vertical flip correction for simulation render

## Future Improvements

1. ~~**Data Augmentation**: Add color jitter, random crops, and perturbations during training~~ (Implemented: static frame augmentation)
2. ~~**Temporal Weighting**: Weight critical timesteps (start, gripper changes) more heavily in loss~~ (Implemented: weighted loss)
3. **Multi-Demo Training**: Train on more demonstrations per task for better coverage
4. **State Matching**: Initialize evaluation from demo-like states to reduce distribution gap
5. **Action Smoothing**: Apply temporal smoothing to predicted action chunks
6. **Larger Policy Head**: Increase transformer capacity (more layers, larger hidden dim)
7. **Different Encoder**: Try fine-tuning V-JEPA 2 or using a different visual encoder

## License

This project uses V-JEPA 2 weights which are subject to Meta's license terms.
