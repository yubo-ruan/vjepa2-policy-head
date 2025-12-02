# V-JEPA 2 Policy v3 Results

## Training Summary

- **Best Validation Loss**: 0.3548 (epoch 80)
- **Final Train Loss**: 0.3780 (68.6% reduction)
- **Final Val Loss**: 0.3570 (69.1% reduction)
- **Epochs**: 100
- **Training Time**: ~15 minutes

## Conservative Configuration (v3)

| Setting | Value |
|---------|-------|
| gripper_oversample | 2 |
| gripper_jitter | 0 (disabled) |
| start_weight | 2.0 |
| transition_weight | 3.0 |
| transition_window | 5 |
| gripper_loss_weight | 1.5 |

## LIBERO Spatial Evaluation

**Overall Success Rate: 8.0%** (5 episodes per task)

| Task | Description | Success |
|------|-------------|---------|
| 1 | bowl between plate & ramekin | 0% |
| 2 | bowl next to ramekin | 0% |
| 3 | bowl from table center | **60%** |
| 4 | bowl on cookie box | **20%** |
| 5 | bowl in cabinet drawer | 0% |
| 6 | bowl on ramekin | 0% |
| 7 | bowl next to cookie box | 0% |
| 8 | bowl on stove | 0% |
| 9 | bowl next to plate | 0% |
| 10 | bowl on wooden cabinet | 0% |

## Files

- `training_results.png` - Training loss curves
- `evaluation_results.png` - Per-task success rates
- `videos/` - Episode recordings as GIFs

## Model Architecture

- V-JEPA 2 ViT-Giant encoder (1.1B params, frozen)
- 64 spatial tokens (8×8 grid) per modality
- 4-layer Transformer decoder
- Action chunking: 50 timesteps
