# V6 Reflection and V7 Planning

## V6 Results Summary

| Metric | Value |
|--------|-------|
| Success Rate | **0%** (50/50 episodes failed) |
| Training Duration | ~100 minutes (200 epochs) |
| Final Train Loss | 0.6325 |
| Final Val Loss | 0.8273 |
| Model Parameters | 60.45M |

## V6 Changes from V3 (Baseline: 8% Success)

| Component | V3 | V6 | Rationale |
|-----------|----|----|-----------|
| Hidden dim | 512 | 768 | More capacity for complex reasoning |
| Num layers | 6 | 6 | Deeper model for better representations |
| Gripper oversample | 2x | 4x | More gripper transition samples |
| Gripper jitter | 0 | 3 | Temporal augmentation around transitions |
| Gripper loss weight | 1.5 | 2.5 | Stronger gripper learning signal |
| Training epochs | 100 | 200 | Longer training for larger model |
| Warmup epochs | 5 | 10 | More gradual warmup |

## Why V6 Failed (0% vs V3's 8%)

### Root Cause Analysis

1. **Overfitting to Gripper Transitions**
   - 92.6% of training samples are gripper transition samples (after 4x oversampling)
   - Model learned "when to grip" but forgot "where to move"
   - Training data became severely imbalanced

2. **Model Capacity vs Data Size Mismatch**
   - V6 has 60M params (58% more than V3)
   - Still only 18K samples
   - Larger models need more data to generalize
   - Overfitting happened faster despite longer training

3. **Loss Convergence Doesn't Equal Policy Quality**
   - V6 achieved lower loss (0.63) than V3 likely did
   - But lower loss on augmented data != better policy
   - Model learned the augmented distribution, not the task

4. **Temporal Jitter Corrupted Causal Relationships**
   - Adding jitter around gripper transitions
   - May have broken the temporal relationship between "reaching" and "grasping"
   - Robot needs to learn: reach → position → close gripper
   - Jitter made this sequence less clear in training data

### Observations from Evaluation Logs

```
Step 0: gripper votes: [1], result: 1.0    # OPEN (correct for start)
Step 50: gripper votes: [-1, -1, 1, 1, 1]  # Mixed votes
Step 100: gripper votes: [1, 1, 1, 1, 1]   # OPEN (should be approaching)
Step 150: gripper votes: [1, 1, 1, 1, 1]   # OPEN (never closes)
Step 200: gripper votes: [1, 1, 1, 1, 1]   # OPEN
Step 250: gripper votes: [1, 1, 1, 1, 1]   # OPEN
```

The gripper is mostly voting OPEN throughout the entire episode, suggesting:
- Position prediction is not moving arm correctly
- Gripper doesn't transition to CLOSE because arm never reaches target

## V7 Plan: Back to Basics with Key Fixes

### Philosophy
Instead of adding complexity, **simplify and debug**. V3 achieved 8% - understand WHY before adding features.

### V7 Configuration

```yaml
# V-JEPA 2 Policy Configuration - v7 Diagnostic

# Model Architecture - EXACT SAME as V3
model:
  hidden_dim: 512              # Back to V3 baseline
  num_layers: 4                # Back to V3 baseline
  num_heads: 8
  dropout: 0.1

# Data - MINIMAL AUGMENTATION
data:
  suite: libero_spatial        # Single suite
  gripper_oversample: 1        # NO oversampling (back to natural distribution)
  gripper_jitter: 0            # NO jitter
  noise_std: 0.02              # Reduce noise (was 0.05)

# Loss - BALANCED
loss:
  gripper_loss_weight: 1.0     # Equal weighting
  start_weight: 1.5            # Slight emphasis on start (was 2.0)
  transition_weight: 2.0       # Less emphasis on transitions (was 3.0)

# Training - STANDARD
training:
  epochs: 100                  # Back to V3 baseline
  lr: 1.0e-4
  warmup_epochs: 5             # Back to V3
```

### Key V7 Changes

1. **Remove Gripper Oversampling**
   - Let natural distribution guide learning
   - Model will see realistic proportion of static vs dynamic frames

2. **Simplify Loss Weighting**
   - Balance position and gripper losses equally
   - Reduce transition focus - model needs to learn full trajectory

3. **Add Position Metrics**
   - Log position prediction error separately
   - Monitor if arm is moving toward target

4. **Debug Mode**
   - Add visualization of predicted trajectories
   - Compare predicted vs actual action sequences

### Alternative Approaches for V8+

If V7 still fails, consider:

1. **Curriculum Learning**
   - Start with static positioning tasks
   - Gradually add gripper operations

2. **Separate Policies**
   - Position policy (6-DoF)
   - Gripper policy (binary)
   - Combine at inference

3. **Behavior Cloning Baseline**
   - Test simple MLP policy on same embeddings
   - If MLP works, problem is in transformer architecture
   - If MLP fails, problem is in embeddings/data

4. **End-to-End Fine-tuning**
   - Unfreeze V-JEPA encoder
   - Let gradients flow back to visual representations
   - Requires more GPU memory but may help

### Success Metrics for V7

| Metric | Target |
|--------|--------|
| Success Rate | ≥8% (match V3) |
| Position MSE | Decreasing during episode |
| Gripper Accuracy | >50% correct transitions |

## Experimental Log

| Version | Success Rate | Key Change | Outcome |
|---------|-------------|------------|---------|
| V3 | 8% | Baseline spatial | Best so far |
| V4 | ? | Goal-conditioned queries | Not evaluated |
| V5 | 0% | Multi-suite training | Gradient conflict |
| V6 | 0% | Larger model + gripper focus | Overfitting |
| V7 | TBD | Simplified, diagnostic | Next experiment |

## Conclusion

V6 demonstrates that **more is not always better**:
- More parameters → more overfitting risk
- More augmentation → more distribution shift
- More gripper focus → less position learning

V7 will return to the V3 baseline and focus on understanding WHY it achieved 8% before attempting improvements.
