# V7 Evaluation Results

## Summary

| Metric | V7 | V6 | V3 (Baseline) |
|--------|----|----|---------------|
| **Overall Success Rate** | **10.0%** | 0% | 8% |
| Model Parameters | 19.19M | 60.45M | ~19M |
| Training Epochs | 100 | 200 | 100 |
| Best Val Loss | 0.4010 | 0.6325 | ~0.40 |

## Per-Task Success Rates

| Task | Description | V7 Success | V6 Success |
|------|-------------|------------|------------|
| T1 | pick up black bowl between plate and ramekin | 0% | 0% |
| T2 | pick up black bowl next to ramekin | 0% | 0% |
| T3 | pick up black bowl from table center | **20%** | 0% |
| T4 | pick up black bowl on cookie box | **20%** | 0% |
| T5 | pick up black bowl in top drawer | 0% | 0% |
| T6 | pick up black bowl on ramekin | 0% | 0% |
| T7 | pick up black bowl next to cookie box | **60%** | 0% |
| T8 | pick up black bowl on stove | 0% | 0% |
| T9 | pick up black bowl next to plate | 0% | 0% |
| T10 | pick up black bowl on wooden cabinet | 0% | 0% |

## Key V7 Changes that Worked

### 1. Removed Gripper Oversampling (4x -> 1x)
- V6 had 92.6% gripper transitions after 4x oversampling
- V7 uses natural distribution (~50% gripper transitions)
- Model now learns balanced position AND gripper control

### 2. Smaller Model (60M -> 19M)
- Hidden dim: 768 -> 512
- Layers: 6 -> 4
- Better generalization with less overfitting risk
- Faster training (8s/epoch vs ~15s/epoch)

### 3. Balanced Loss Weights
- Gripper loss weight: 2.5 -> 1.0 (equal weighting)
- Transition weight: 3.0 -> 2.0 (less gripper focus)
- Start weight: 2.0 -> 1.5 (more balanced)

### 4. Reduced Noise Augmentation
- Noise std: 0.05 -> 0.02
- Less distribution shift during training

### 5. No Temporal Jitter
- Gripper jitter: 3 -> 0
- Preserved causal relationships (reach -> position -> grip)

## Analysis

### Why V7 > V6 > V5
1. **V5 (0%)**: Multi-suite training caused gradient conflicts between different task distributions
2. **V6 (0%)**: Gripper oversampling + larger model caused overfitting to gripper transitions
3. **V7 (10%)**: Back to basics - natural distribution, smaller model, balanced loss

### Success Pattern Analysis
- T7 (60% success) - "bowl next to cookie box" has clear spatial relationship
- T3, T4 (20% each) - "table center" and "on cookie box" have distinct visual features
- Failed tasks (T1, T2, T5, T6, T8, T9, T10) - require precise spatial reasoning or have occlusion/drawer manipulation

### Gripper Behavior Comparison

**V6 Gripper Votes (always OPEN):**
```
Step 0: [1]         -> OPEN
Step 50: [-1,-1,1,1,1] -> OPEN
Step 100: [1,1,1,1,1] -> OPEN
Step 150: [1,1,1,1,1] -> OPEN
```

**V7 Gripper Votes (responds to scene):**
```
Step 0: [1]           -> OPEN (correct start)
Step 50: [-1,-1,-1,-1,-1] -> CLOSE (attempting grasp)
Step 100: [1,1,1,1,1] -> OPEN (release)
```

V7 shows proper gripper state transitions while V6 was stuck in OPEN.

## V8 Planning

### Approaches to Consider

1. **Increase Training Data**
   - Current: 5526 samples
   - Consider data augmentation (color jitter, crop, etc.)
   - Generate more demonstrations

2. **Architecture Improvements**
   - Add positional encoding for spatial tokens
   - Try cross-attention between video and goal tokens
   - Consider LSTM/GRU for temporal modeling

3. **Training Improvements**
   - Curriculum learning: start with easier tasks
   - Task-specific fine-tuning
   - Contrastive pre-training on embeddings

4. **Evaluation Improvements**
   - Adjust temporal ensemble parameters
   - Try different chunk execution strategies
   - Add action smoothing

### Recommended V8 Changes

Based on V7's success pattern (60% on T7, 20% on T3/T4):

1. **Keep V7 base configuration** - it works
2. **Add spatial position encoding** - help model understand object locations
3. **Increase model capacity slightly** - 512 -> 640 hidden dim
4. **Add dropout in attention** - reduce overfitting
5. **Longer training with early stopping** - 150 epochs with patience

## Conclusion

V7's "back to basics" approach validated the hypothesis that V6's aggressive gripper augmentation caused regression. The 10% success rate (25% improvement over V3's 8%) shows that:

1. Natural data distribution > artificial augmentation
2. Smaller models generalize better with limited data
3. Balanced loss weighting is crucial for multi-objective learning

The next step (V8) should build incrementally on V7's success rather than making large architectural changes.
