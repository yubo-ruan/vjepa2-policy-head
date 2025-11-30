#!/usr/bin/env python3
"""
Visualize training results for V-JEPA 2 Policy v3.

Creates training curves and evaluation results plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Training history from v3 run
epochs = list(range(101))
train_losses = [
    1.2044, 0.9565, 0.9276, 0.9121, 0.8396, 0.7953, 0.7854, 0.7445, 0.7352, 0.7227,
    0.6934, 0.6929, 0.6763, 0.6651, 0.6513, 0.6609, 0.6476, 0.6206, 0.6478, 0.6422,
    0.6201, 0.6261, 0.6116, 0.6081, 0.5976, 0.6035, 0.5771, 0.5947, 0.5866, 0.5823,
    0.5767, 0.5731, 0.5673, 0.5665, 0.5650, 0.5672, 0.5607, 0.5537, 0.5519, 0.5541,
    0.5423, 0.5403, 0.5365, 0.5330, 0.5351, 0.5292, 0.5224, 0.5187, 0.5224, 0.5188,
    0.5089, 0.5111, 0.4978, 0.4974, 0.4995, 0.4939, 0.4958, 0.4889, 0.4837, 0.4782,
    0.4722, 0.4684, 0.4667, 0.4672, 0.4584, 0.4615, 0.4532, 0.4479, 0.4505, 0.4445,
    0.4410, 0.4382, 0.4340, 0.4269, 0.4309, 0.4226, 0.4218, 0.4149, 0.4169, 0.4068,
    0.4082, 0.4017, 0.4035, 0.3993, 0.3974, 0.3966, 0.3908, 0.3891, 0.3906, 0.3902,
    0.3883, 0.3866, 0.3860, 0.3836, 0.3865, 0.3855, 0.3840, 0.3819, 0.3807, 0.3794,
    0.3780
]

val_losses = [
    1.1536, 0.6613, 0.6903, 0.6687, 0.6169, 0.5305, 0.5141, 0.4927, 0.6188, 0.4992,
    0.4837, 0.4710, 0.5672, 0.4848, 0.4842, 0.4931, 0.4800, 0.4985, 0.4980, 0.4472,
    0.4386, 0.4527, 0.5113, 0.4585, 0.4458, 0.4226, 0.4829, 0.4292, 0.4540, 0.4612,
    0.4377, 0.4680, 0.4532, 0.4182, 0.4086, 0.4747, 0.4201, 0.4383, 0.4376, 0.4278,
    0.4335, 0.3924, 0.4059, 0.4182, 0.4008, 0.4393, 0.4083, 0.3962, 0.3813, 0.4072,
    0.3771, 0.3863, 0.3860, 0.4181, 0.3969, 0.3869, 0.3954, 0.3732, 0.3745, 0.3760,
    0.4002, 0.3711, 0.3760, 0.3953, 0.3861, 0.3839, 0.3630, 0.3762, 0.3667, 0.3730,
    0.3721, 0.3810, 0.3610, 0.3806, 0.3724, 0.3723, 0.3634, 0.3778, 0.3689, 0.3700,
    0.3548, 0.3750, 0.3679, 0.3602, 0.3705, 0.3660, 0.3635, 0.3670, 0.3617, 0.3634,
    0.3684, 0.3656, 0.3594, 0.3652, 0.3675, 0.3640, 0.3628, 0.3612, 0.3598, 0.3585,
    0.3570
]

# Evaluation results
task_names = [
    'bowl between\nplate & ramekin',
    'bowl next to\nramekin',
    'bowl from\ntable center',
    'bowl on\ncookie box',
    'bowl in\ncabinet drawer',
    'bowl on\nramekin',
    'bowl next to\ncookie box',
    'bowl on\nstove',
    'bowl next to\nplate',
    'bowl on\nwooden cabinet'
]
success_rates = [0.0, 0.0, 60.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Training curves
ax1 = axes[0, 0]
ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('V-JEPA 2 Policy v3 Training', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)
ax1.set_ylim(0.3, 1.3)

# Mark best validation loss
best_epoch = np.argmin(val_losses)
best_val = val_losses[best_epoch]
ax1.scatter([best_epoch], [best_val], s=100, c='green', marker='*', zorder=5, label=f'Best: {best_val:.4f} (epoch {best_epoch})')
ax1.annotate(f'Best: {best_val:.4f}', xy=(best_epoch, best_val), xytext=(best_epoch+5, best_val+0.1),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

# 2. Per-task success rates
ax2 = axes[0, 1]
colors = ['green' if r > 0 else 'lightcoral' for r in success_rates]
bars = ax2.bar(range(len(success_rates)), success_rates, color=colors, edgecolor='black')
ax2.set_xlabel('Task', fontsize=12)
ax2.set_ylabel('Success Rate (%)', fontsize=12)
ax2.set_title('LIBERO Spatial Task Success Rates', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(task_names)))
ax2.set_xticklabels([f'T{i+1}' for i in range(len(task_names))], fontsize=10)
ax2.set_ylim(0, 100)
ax2.axhline(y=8.0, color='blue', linestyle='--', linewidth=2, label=f'Overall: 8.0%')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, rate in zip(bars, success_rates):
    if rate > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Loss reduction over time
ax3 = axes[1, 0]
train_reduction = [(train_losses[0] - t) / train_losses[0] * 100 for t in train_losses]
val_reduction = [(val_losses[0] - v) / val_losses[0] * 100 for v in val_losses]
ax3.fill_between(epochs, train_reduction, alpha=0.3, color='blue', label='Train Loss Reduction')
ax3.fill_between(epochs, val_reduction, alpha=0.3, color='red', label='Val Loss Reduction')
ax3.plot(epochs, train_reduction, 'b-', linewidth=2)
ax3.plot(epochs, val_reduction, 'r-', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss Reduction (%)', fontsize=12)
ax3.set_title('Training Progress (% Loss Reduction)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 100)

# 4. Model configuration summary
ax4 = axes[1, 1]
ax4.axis('off')
config_text = """
V-JEPA 2 Policy v3 - Conservative Configuration

Model Architecture:
  • V-JEPA 2 ViT-Giant encoder (1.1B params, frozen)
  • 64 spatial tokens (8×8 grid) per modality
  • 4-layer Transformer decoder policy head
  • 8 attention heads, 512 hidden dim
  • Action chunking: 50 timesteps

Training Configuration:
  • Epochs: 100
  • Batch size: 32
  • Learning rate: 1e-4 (cosine schedule)
  • Warmup: 5 epochs

Conservative Loss Settings:
  • Start weight: 2.0 (first 10 steps)
  • Transition weight: 3.0 (5-step window)
  • Gripper loss weight: 1.5
  • Gripper oversample: 2×

Static Augmentation:
  • Static video prob: 25%
  • Beginning-biased prob: 25%

Results:
  • Best val loss: {:.4f} (epoch {})
  • Train loss: {:.4f} → {:.4f} ({:.1f}% reduction)
  • Val loss: {:.4f} → {:.4f} ({:.1f}% reduction)
  • LIBERO Spatial: 8.0% success rate
""".format(
    best_val, best_epoch,
    train_losses[0], train_losses[-1], (1 - train_losses[-1]/train_losses[0])*100,
    val_losses[0], val_losses[-1], (1 - val_losses[-1]/val_losses[0])*100
)
ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_dir = Path('/workspace/vjepa2-policy-head/results/v3')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'training_results.png', dpi=150, bbox_inches='tight')
print(f"Saved training visualization to: {output_dir / 'training_results.png'}")

# Also create a simpler summary figure
fig2, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(task_names)), success_rates, color=colors, edgecolor='black', width=0.7)
ax.set_xlabel('Task Description', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('V-JEPA 2 Policy v3: LIBERO Spatial Evaluation\n(5 episodes per task)', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(task_names)))
ax.set_xticklabels(task_names, fontsize=8, rotation=45, ha='right')
ax.set_ylim(0, 100)
ax.axhline(y=8.0, color='blue', linestyle='--', linewidth=2, label=f'Overall: 8.0%')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (rate, bar) in enumerate(zip(success_rates, ax.patches)):
    if rate > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{rate:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'evaluation_results.png', dpi=150, bbox_inches='tight')
print(f"Saved evaluation visualization to: {output_dir / 'evaluation_results.png'}")

plt.show()
print("\nVisualization complete!")
