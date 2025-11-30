"""
Full Diagnostic Report for V-JEPA 2 Policy

Tests:
1. Embedding Consistency - precomputed vs live
2. Action Variance - detect prediction collapse
3. Train vs Eval State Distance - distribution shift
4. Prediction Quality on Training States
5. Prediction Quality on Eval States

This will identify the root cause of 0% success.
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2PolicySpatial
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load trained spatial model"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    n_spatial_tokens = config.get('spatial', {}).get('n_tokens', 64)

    model = VJEPA2PolicySpatial(
        vjepa2_model_path=config['vjepa2']['model_path'],
        vjepa2_model_name=config['vjepa2']['model_name'],
        vjepa2_freeze=True,
        vjepa2_num_frames=config['vjepa2']['num_frames'],
        proprio_dim=config['proprio']['dim'],
        proprio_history=config['proprio']['history_len'],
        proprio_output_dim=config['proprio'].get('output_dim', 256),
        policy_hidden_dim=config['policy']['hidden_dim'],
        policy_n_heads=config['policy']['n_heads'],
        policy_n_layers=config['policy']['n_layers'],
        n_spatial_tokens=n_spatial_tokens,
        n_proprio_tokens=config['policy'].get('n_context_tokens', 4),
        action_dim=config['policy']['action_dim'],
        chunk_size=config['policy']['chunk_size'],
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config


def test_embedding_consistency(model, demo_file, device="cuda"):
    """Test 1: Compare precomputed embeddings to live encoding"""
    print("\n" + "=" * 60)
    print("[TEST 1] EMBEDDING CONSISTENCY")
    print("=" * 60)

    # Load precomputed embeddings
    precomputed_dir = Path("/workspace/data/embeddings_spatial/libero_spatial")
    precomputed_files = sorted(precomputed_dir.glob("*.pt"))

    if not precomputed_files:
        print("No precomputed embeddings found!")
        return None

    # Load first precomputed sample
    precomputed = torch.load(precomputed_files[0])
    precomputed_video = precomputed['video_tokens']

    print(f"Precomputed video tokens shape: {precomputed_video.shape}")
    print(f"Precomputed from: {precomputed_files[0].name}")

    # Get corresponding raw data and encode live
    # Parse filename to get demo info
    filename = precomputed_files[0].stem  # e.g., "task_0_demo_0_sample_0"
    parts = filename.split('_')

    # Load raw demo
    with h5py.File(demo_file, 'r') as f:
        # Get first 16 frames from demo 0
        for key in ['agentview_rgb', 'agentview_image']:
            full_key = 'data/demo_0/obs/' + key
            if full_key in f:
                images = f[full_key][:16]
                break
        else:
            print("Could not find images in demo file")
            return None

    # Encode live
    video_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor.unsqueeze(0).to(device)

    # Resize to 256x256 if needed
    if video_tensor.shape[-1] != 256:
        B, T, C, H, W = video_tensor.shape
        video_tensor = video_tensor.view(B * T, C, H, W)
        video_tensor = F.interpolate(video_tensor, size=(256, 256), mode='bilinear')
        video_tensor = video_tensor.view(B, T, C, 256, 256)

    with torch.no_grad():
        live_video = model.vjepa2.encode_video_spatial(video_tensor)
        live_video = F.normalize(live_video, dim=-1)

    # Also normalize precomputed for fair comparison
    precomputed_video_norm = F.normalize(precomputed_video.to(device), dim=-1)

    # Compare
    cos_sim = F.cosine_similarity(
        precomputed_video_norm.flatten().unsqueeze(0),
        live_video.flatten().unsqueeze(0)
    ).item()

    l2_dist = (precomputed_video_norm - live_video).pow(2).mean().sqrt().item()

    print(f"\nPrecomputed vs Live:")
    print(f"  Cosine Similarity: {cos_sim:.6f}")
    print(f"  L2 Distance: {l2_dist:.6f}")

    if cos_sim > 0.99:
        print("✓ Embeddings are consistent!")
        return "pass"
    elif cos_sim > 0.9:
        print("⚠ Embeddings are somewhat similar but not identical")
        return "warning"
    else:
        print("✗ EMBEDDINGS MISMATCH! This is likely the bug!")
        return "fail"


def test_action_variance(model, n_samples=10, device="cuda"):
    """Test 2: Check if model predicts different actions for different inputs"""
    print("\n" + "=" * 60)
    print("[TEST 2] ACTION VARIANCE (Collapse Detection)")
    print("=" * 60)

    # Load precomputed embeddings
    precomputed_dir = Path("/workspace/data/embeddings_spatial/libero_spatial")
    precomputed_files = sorted(precomputed_dir.glob("*.pt"))[:n_samples]

    if len(precomputed_files) < n_samples:
        print(f"Only found {len(precomputed_files)} precomputed samples")
        n_samples = len(precomputed_files)

    all_preds = []

    for f in precomputed_files:
        sample = torch.load(f)

        with torch.no_grad():
            pred = model.forward_with_precomputed(
                sample['video_tokens'].unsqueeze(0).to(device),
                sample['goal_tokens'].unsqueeze(0).to(device),
                sample['proprio'].unsqueeze(0).to(device),
            )

        all_preds.append(pred[0, 0].cpu())

    all_preds = torch.stack(all_preds)
    variance_per_dim = all_preds.var(dim=0)
    mean_variance = variance_per_dim.mean().item()

    print(f"\nAction predictions across {n_samples} samples:")
    print(f"  Per-dim variance: {variance_per_dim.numpy()}")
    print(f"  Mean variance: {mean_variance:.6f}")

    # Also check the range of predictions
    pred_min = all_preds.min(dim=0).values.numpy()
    pred_max = all_preds.max(dim=0).values.numpy()
    pred_range = pred_max - pred_min

    print(f"\n  Per-dim range: {pred_range}")
    print(f"  Mean range: {pred_range.mean():.4f}")

    if mean_variance < 0.001:
        print("\n✗ ACTIONS COLLAPSED! Model predicts nearly same action for all inputs!")
        return "fail"
    elif mean_variance < 0.01:
        print("\n⚠ Low action variance - model may be under-differentiating")
        return "warning"
    else:
        print("\n✓ Actions vary across inputs")
        return "pass"


def test_prediction_quality_on_training(model, n_samples=10, device="cuda"):
    """Test 4: Check prediction quality on training data"""
    print("\n" + "=" * 60)
    print("[TEST 4] PREDICTION QUALITY ON TRAINING DATA")
    print("=" * 60)

    precomputed_dir = Path("/workspace/data/embeddings_spatial/libero_spatial")
    precomputed_files = sorted(precomputed_dir.glob("*.pt"))[:n_samples]

    mses = []
    cos_sims = []
    per_dim_errors = []

    for f in precomputed_files:
        sample = torch.load(f)

        with torch.no_grad():
            pred = model.forward_with_precomputed(
                sample['video_tokens'].unsqueeze(0).to(device),
                sample['goal_tokens'].unsqueeze(0).to(device),
                sample['proprio'].unsqueeze(0).to(device),
            )

        pred_action = pred[0, 0].cpu().numpy()
        gt_action = sample['actions'][0].numpy()

        mse = ((pred_action - gt_action) ** 2).mean()
        cos = np.dot(pred_action, gt_action) / (
            np.linalg.norm(pred_action) * np.linalg.norm(gt_action) + 1e-8
        )
        dim_error = np.abs(pred_action - gt_action)

        mses.append(mse)
        cos_sims.append(cos)
        per_dim_errors.append(dim_error)

    per_dim_errors = np.array(per_dim_errors)
    mean_dim_errors = per_dim_errors.mean(axis=0)

    print(f"\nOn {n_samples} training samples:")
    print(f"  Mean MSE: {np.mean(mses):.6f}")
    print(f"  Mean Cosine Similarity: {np.mean(cos_sims):.4f}")
    print(f"  Per-dim mean absolute error:")

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'grip']
    for i, name in enumerate(dim_names):
        print(f"    {name}: {mean_dim_errors[i]:.4f}")

    if np.mean(cos_sims) > 0.9:
        print("\n✓ Excellent predictions on training data!")
        return "pass"
    elif np.mean(cos_sims) > 0.7:
        print("\n⚠ Good predictions on training data")
        return "warning"
    elif np.mean(cos_sims) > 0.5:
        print("\n⚠ Moderate predictions on training data")
        return "warning"
    else:
        print("\n✗ POOR PREDICTIONS EVEN ON TRAINING DATA!")
        return "fail"


def test_eval_vs_training_distance(model, task, bddl_file_path, device="cuda"):
    """Test 3 & 5: Compare eval state to training states"""
    print("\n" + "=" * 60)
    print("[TEST 3 & 5] EVAL vs TRAINING STATE COMPARISON")
    print("=" * 60)

    # Create eval environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        render_camera="agentview",
        camera_heights=256,
        camera_widths=256,
    )

    obs = env.reset()
    eval_image = obs['agentview_image']

    # Encode eval state
    eval_tensor = torch.from_numpy(eval_image).permute(2, 0, 1).float() / 255.0
    eval_tensor = eval_tensor.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and time dims

    # Repeat to make 16 frames
    eval_tensor = eval_tensor.repeat(1, 16, 1, 1, 1)

    with torch.no_grad():
        eval_emb = model.vjepa2.encode_video_spatial(eval_tensor)
        eval_emb = F.normalize(eval_emb, dim=-1)

    # Load training embeddings and find closest
    precomputed_dir = Path("/workspace/data/embeddings_spatial/libero_spatial")
    precomputed_files = sorted(precomputed_dir.glob("*.pt"))[:100]

    min_distance = float('inf')
    max_cos_sim = -1
    closest_idx = -1
    closest_sample = None

    train_embeddings = []

    for i, f in enumerate(precomputed_files):
        sample = torch.load(f)
        train_emb = F.normalize(sample['video_tokens'].to(device), dim=-1)
        train_embeddings.append(train_emb)

        # L2 distance
        dist = (eval_emb - train_emb).pow(2).mean().sqrt().item()

        # Cosine similarity (flatten to compare overall)
        cos = F.cosine_similarity(
            eval_emb.flatten().unsqueeze(0),
            train_emb.flatten().unsqueeze(0)
        ).item()

        if dist < min_distance:
            min_distance = dist
            max_cos_sim = cos
            closest_idx = i
            closest_sample = sample

    print(f"\nEval state comparison to {len(precomputed_files)} training samples:")
    print(f"  Closest sample index: {closest_idx}")
    print(f"  Min L2 distance: {min_distance:.4f}")
    print(f"  Max cosine similarity: {max_cos_sim:.4f}")

    # Compute train-train distances for reference
    train_distances = []
    for i in range(min(10, len(train_embeddings))):
        for j in range(i + 1, min(10, len(train_embeddings))):
            d = (train_embeddings[i] - train_embeddings[j]).pow(2).mean().sqrt().item()
            train_distances.append(d)

    avg_train_dist = np.mean(train_distances) if train_distances else 0
    print(f"  Avg train-train distance: {avg_train_dist:.4f}")
    print(f"  Ratio (eval-train / train-train): {min_distance / (avg_train_dist + 1e-8):.2f}")

    # Test prediction at eval state
    print("\n--- Prediction at eval state ---")

    # Get proprio from eval
    proprio_parts = []
    if 'robot0_eef_pos' in obs:
        proprio_parts.append(obs['robot0_eef_pos'])
    if 'robot0_eef_quat' in obs:
        # Convert quat to euler
        quat = obs['robot0_eef_quat']
        x, y, z, w = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        proprio_parts.append(np.array([roll, pitch, yaw]))
    if 'robot0_gripper_qpos' in obs:
        proprio_parts.append(obs['robot0_gripper_qpos'])
    if 'robot0_joint_pos' in obs:
        proprio_parts.append(obs['robot0_joint_pos'])

    eval_proprio = np.concatenate(proprio_parts)[:15]
    eval_proprio = np.tile(eval_proprio, (5, 1))  # 5 history steps
    eval_proprio_tensor = torch.from_numpy(eval_proprio).float().unsqueeze(0).to(device)

    # Get prediction using closest sample's goal
    with torch.no_grad():
        eval_pred = model.forward_with_precomputed(
            eval_emb,
            closest_sample['goal_tokens'].unsqueeze(0).to(device),
            eval_proprio_tensor,
        )

    eval_action = eval_pred[0, 0].cpu().numpy()
    closest_action = closest_sample['actions'][0].numpy()

    cos_sim = np.dot(eval_action, closest_action) / (
        np.linalg.norm(eval_action) * np.linalg.norm(closest_action) + 1e-8
    )

    print(f"\nEval prediction vs closest training action:")
    print(f"  Cosine similarity: {cos_sim:.4f}")

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'grip']
    print(f"\n  {'Dim':<8} {'Eval':<12} {'Closest':<12} {'Diff':<12}")
    print("  " + "-" * 44)
    for i, name in enumerate(dim_names):
        diff = eval_action[i] - closest_action[i]
        print(f"  {name:<8} {eval_action[i]:<12.4f} {closest_action[i]:<12.4f} {diff:<12.4f}")

    env.close()

    if min_distance > 2 * avg_train_dist:
        print("\n✗ EVAL STATE IS FAR FROM TRAINING DISTRIBUTION!")
        return "distribution_shift"
    elif cos_sim < 0.5:
        print("\n✗ PREDICTIONS DON'T TRANSFER TO EVAL STATES!")
        return "generalization_failure"
    else:
        print("\n✓ Eval state within distribution and predictions transfer")
        return "pass"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/workspace/checkpoints_spatial/best_model.pt')
    parser.add_argument('--config', type=str, default='configs/spatial.yaml')
    parser.add_argument('--suite', type=str, default='libero_spatial')
    args = parser.parse_args()

    print("=" * 70)
    print("FULL DIAGNOSTIC REPORT")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, args.config, device)

    # Get benchmark task
    BenchClass = benchmark.get_benchmark(args.suite)
    bench = BenchClass()
    task = bench.get_task(0)
    bddl_file_path = bench.get_task_bddl_file_path(0)

    # Get demo file
    demo_dir = Path("/workspace/data/libero") / task.problem_folder
    demo_file = demo_dir / f"{task.name}_demo.hdf5"

    results = {}

    # Run tests
    results['embedding_consistency'] = test_embedding_consistency(model, demo_file, device)
    results['action_variance'] = test_action_variance(model, n_samples=20, device=device)
    results['prediction_on_training'] = test_prediction_quality_on_training(model, n_samples=20, device=device)
    results['eval_vs_training'] = test_eval_vs_training_distance(model, task, bddl_file_path, device)

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status = "✓" if result == "pass" else ("⚠" if result == "warning" else "✗")
        print(f"  {status} {test_name}: {result}")

    # Determine root cause
    print("\n" + "-" * 70)
    print("DIAGNOSIS:")
    print("-" * 70)

    if results['embedding_consistency'] == 'fail':
        print("""
ROOT CAUSE: EMBEDDING MISMATCH

The precomputed embeddings used for training don't match live encoding.
This means the model learned to predict actions for DIFFERENT embeddings
than what it receives during evaluation.

FIX: Re-precompute embeddings with exactly the same encoding pipeline
used during evaluation, including:
- Same image preprocessing (resize, normalize)
- Same frame sampling
- Same normalization (L2 normalize or not)
""")
    elif results['action_variance'] == 'fail':
        print("""
ROOT CAUSE: ACTION COLLAPSE

The model predicts nearly the same action regardless of input.
This is a training failure - the model hasn't learned to differentiate states.

FIX:
- Check for bugs in training loop
- Increase model capacity
- Use stronger augmentation to prevent overfitting to mean
""")
    elif results['prediction_on_training'] == 'fail':
        print("""
ROOT CAUSE: POOR TRAINING FIT

The model doesn't even predict well on training data.
This suggests the model hasn't converged or there's a bug.

FIX:
- Train for more epochs
- Check loss is decreasing
- Verify data loading is correct
""")
    elif results['eval_vs_training'] in ['distribution_shift', 'generalization_failure']:
        print("""
ROOT CAUSE: GENERALIZATION FAILURE

The model works on training data but fails on evaluation because:
1. Eval initial states are different from training states
2. Model hasn't learned generalizable features

This is the fundamental behavior cloning limitation.

FIX (in order of effort):
1. Heavy data augmentation (color jitter, crops, noise)
2. Train on more diverse data (all LIBERO suites)
3. Use techniques like DAgger for interactive learning
4. Your research direction: latent subgoals for higher-level abstraction
""")
    else:
        print("""
All tests passed but evaluation still fails.
This is unexpected - further investigation needed.
""")


if __name__ == "__main__":
    main()
