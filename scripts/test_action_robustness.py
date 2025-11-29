"""
Test action robustness - analyze how model predictions compare to ground truth.

This script:
1. Loads model and demos
2. For each demo frame, predicts actions and compares to GT
3. Checks if predictions are "close enough" to work
4. Tests if adding noise to GT actions still succeeds
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


def compare_actions_on_demo(
    model,
    task,
    bddl_file_path: str,
    demo_dir: str = "/workspace/data/libero",
    demo_idx: int = 0,
    device: str = "cuda",
    verbose: bool = True,
):
    """Compare model predictions to ground truth actions along demo trajectory"""

    suite_dir = Path(demo_dir) / task.problem_folder
    demo_file = suite_dir / f"{task.name}_demo.hdf5"

    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        return None

    # Load demo data
    with h5py.File(demo_file, 'r') as f:
        actions = f[f'data/demo_{demo_idx}/actions'][:]
        states = f[f'data/demo_{demo_idx}/states'][:]

        # Try to get images
        for key in ['agentview_rgb', 'agentview_image']:
            full_key = f'data/demo_{demo_idx}/obs/{key}'
            if full_key in f:
                images = f[full_key][:]
                break
        else:
            print("No images found in demo")
            return None

        # Get goal image (last frame)
        goal_image = images[-1]

        # Get proprio data
        proprio_keys = ['ee_pos', 'ee_states', 'gripper_states', 'joint_states']
        proprio_data = []
        for key in proprio_keys:
            full_key = f'data/demo_{demo_idx}/obs/{key}'
            if full_key in f:
                proprio_data.append(f[full_key][:])
        if proprio_data:
            proprio = np.concatenate(proprio_data, axis=-1)[:, :15]
        else:
            print("No proprio data found")
            return None

    print(f"Demo: {len(actions)} steps, images: {images.shape}, proprio: {proprio.shape}")

    # Compare predictions at different timesteps
    timesteps_to_check = [0, len(actions)//4, len(actions)//2, 3*len(actions)//4]

    results = {
        'position_errors': [],
        'rotation_errors': [],
        'gripper_errors': [],
    }

    for t in timesteps_to_check:
        if t >= len(actions) - 16:
            continue

        # Build video from t to t+16
        video_frames = images[max(0, t-15):t+1]
        if len(video_frames) < 16:
            video_frames = np.concatenate([
                np.tile(video_frames[:1], (16 - len(video_frames), 1, 1, 1)),
                video_frames
            ], axis=0)

        # Build proprio history
        proprio_history = proprio[max(0, t-4):t+1]
        if len(proprio_history) < 5:
            proprio_history = np.concatenate([
                np.tile(proprio_history[:1], (5 - len(proprio_history), 1)),
                proprio_history
            ], axis=0)

        # Prepare tensors
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0).to(device)

        # Resize to 256x256 if needed
        if video_tensor.shape[-1] != 256:
            B, T, C, H, W = video_tensor.shape
            video_tensor = video_tensor.view(B * T, C, H, W)
            video_tensor = F.interpolate(video_tensor, size=(256, 256), mode='bilinear')
            video_tensor = video_tensor.view(B, T, C, 256, 256)

        goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
        goal_tensor = goal_tensor.unsqueeze(0).to(device)
        if goal_tensor.shape[-1] != 256:
            goal_tensor = F.interpolate(goal_tensor, size=(256, 256), mode='bilinear')

        proprio_tensor = torch.from_numpy(proprio_history).float().unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            # Encode video and goal
            video_tokens = model.vjepa2.encode_video_spatial(video_tensor)
            goal_tokens = model.vjepa2.encode_image_spatial(goal_tensor)

            # Normalize
            video_tokens = F.normalize(video_tokens, dim=-1)
            goal_tokens = F.normalize(goal_tokens, dim=-1)

            # Get action prediction
            pred_actions = model.forward_with_precomputed(video_tokens, goal_tokens, proprio_tensor)
            pred_action = pred_actions[0, 0].cpu().numpy()  # First action of chunk

        # Compare to GT
        gt_action = actions[t]

        pos_error = np.abs(pred_action[:3] - gt_action[:3]).mean()
        rot_error = np.abs(pred_action[3:6] - gt_action[3:6]).mean()
        grip_error = np.abs(pred_action[6] - gt_action[6])

        results['position_errors'].append(pos_error)
        results['rotation_errors'].append(rot_error)
        results['gripper_errors'].append(grip_error)

        if verbose:
            print(f"\nTimestep {t}:")
            print(f"  GT action:   pos={gt_action[:3]}, rot={gt_action[3:6]}, grip={gt_action[6]:.2f}")
            print(f"  Pred action: pos={pred_action[:3]}, rot={pred_action[3:6]}, grip={pred_action[6]:.2f}")
            print(f"  Errors: pos={pos_error:.4f}, rot={rot_error:.4f}, grip={grip_error:.4f}")

    return results


def test_noisy_gt_actions(
    task,
    bddl_file_path: str,
    demo_dir: str = "/workspace/data/libero",
    demo_idx: int = 0,
    noise_levels: list = [0.0, 0.01, 0.05, 0.1, 0.2],
):
    """Test if GT actions still work with added noise"""

    suite_dir = Path(demo_dir) / task.problem_folder
    demo_file = suite_dir / f"{task.name}_demo.hdf5"

    with h5py.File(demo_file, 'r') as f:
        actions = f[f'data/demo_{demo_idx}/actions'][:]
        states = f[f'data/demo_{demo_idx}/states'][:]

    results = {}

    for noise_std in noise_levels:
        print(f"\n--- Testing noise_std = {noise_std} ---")

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            render_camera="agentview",
            camera_heights=256,
            camera_widths=256,
        )

        # Reset and restore initial state
        env.reset()
        try:
            env.env.sim.set_state_from_flattened(states[0])
            env.env.sim.forward()
        except:
            env.close()
            continue

        # Add noise to actions
        noisy_actions = actions.copy()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, actions.shape)
            noisy_actions[:, :6] += noise[:, :6]  # Position and rotation
            noisy_actions = np.clip(noisy_actions, -1, 1)

        # Replay
        success = False
        total_reward = 0

        for step, action in enumerate(noisy_actions):
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                if hasattr(env.env, '_check_success'):
                    success = env.env._check_success()
                break

        env.close()

        results[noise_std] = {
            'success': success,
            'total_reward': total_reward,
        }

        status = "SUCCESS" if success else "FAIL"
        print(f"  Result: {status}, reward: {total_reward:.3f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/workspace/checkpoints_spatial/best_model.pt')
    parser.add_argument('--config', type=str, default='configs/spatial.yaml')
    parser.add_argument('--suite', type=str, default='libero_spatial')
    parser.add_argument('--n_tasks', type=int, default=2)
    args = parser.parse_args()

    print("="*60)
    print("ACTION ROBUSTNESS ANALYSIS")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, args.config, device)

    # Get benchmark
    BenchClass = benchmark.get_benchmark(args.suite)
    bench = BenchClass()

    all_pos_errors = []
    all_rot_errors = []
    all_grip_errors = []

    for task_idx in range(min(args.n_tasks, bench.get_num_tasks())):
        task = bench.get_task(task_idx)
        bddl_file_path = bench.get_task_bddl_file_path(task_idx)
        task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"

        print(f"\n{'='*60}")
        print(f"Task {task_idx}: {task_name[:50]}...")
        print(f"{'='*60}")

        # Compare predictions to GT
        print("\n--- Comparing model predictions to ground truth ---")
        results = compare_actions_on_demo(
            model, task, bddl_file_path, device=device, verbose=True
        )

        if results:
            all_pos_errors.extend(results['position_errors'])
            all_rot_errors.extend(results['rotation_errors'])
            all_grip_errors.extend(results['gripper_errors'])

        # Test noisy GT actions
        print("\n--- Testing noise tolerance ---")
        noise_results = test_noisy_gt_actions(
            task, bddl_file_path, noise_levels=[0.0, 0.05, 0.1, 0.2]
        )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if all_pos_errors:
        print(f"\nMean prediction errors:")
        print(f"  Position: {np.mean(all_pos_errors):.4f} (std: {np.std(all_pos_errors):.4f})")
        print(f"  Rotation: {np.mean(all_rot_errors):.4f} (std: {np.std(all_rot_errors):.4f})")
        print(f"  Gripper:  {np.mean(all_grip_errors):.4f} (std: {np.std(all_grip_errors):.4f})")

        # Interpretation
        if np.mean(all_pos_errors) > 0.3:
            print("\n>>> Position errors are HIGH (>0.3)")
            print("    Model predictions are far from ground truth")
            print("    Consider more training data or different architecture")
        elif np.mean(all_pos_errors) > 0.1:
            print("\n>>> Position errors are MODERATE (0.1-0.3)")
            print("    Model partially captures trajectory but may drift")
            print("    Receding horizon and data augmentation may help")
        else:
            print("\n>>> Position errors are LOW (<0.1)")
            print("    Model predictions are close to ground truth!")
            print("    Issue may be in state-action mismatch during eval")


if __name__ == "__main__":
    main()
