#!/usr/bin/env python3
"""
V7 Evaluation Script with Video Recording

Records GIFs of all episodes for all tasks.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
from collections import deque
import imageio

# Fix for LIBERO's torch.load calls with numpy arrays
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.policy import VJEPA2Policy

# LIBERO imports
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

VALID_SUITES = ['libero_spatial', 'libero_object', 'libero_goal']


def quat_to_euler(quat):
    """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw]"""
    x, y, z, w = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def extract_proprio(obs):
    """Extract proprioception from observation

    Must match training format: ee_pos(3) + ee_ori(3) + gripper(2) + joint_pos(7) = 15 dims
    """
    proprio_parts = []

    # EEF position (3D)
    if 'robot0_eef_pos' in obs:
        proprio_parts.append(obs['robot0_eef_pos'])

    # EEF orientation: convert quaternion (4D) to euler (3D)
    if 'robot0_eef_quat' in obs:
        quat = obs['robot0_eef_quat']
        euler = quat_to_euler(quat)
        proprio_parts.append(euler)

    # Gripper state (2D)
    if 'robot0_gripper_qpos' in obs:
        proprio_parts.append(obs['robot0_gripper_qpos'])

    # Joint positions (7D)
    if 'robot0_joint_pos' in obs:
        proprio_parts.append(obs['robot0_joint_pos'])

    return np.concatenate(proprio_parts) if proprio_parts else np.zeros(15)


def run_episode_with_video(
    model,
    env,
    goal_emb,
    device='cuda',
    video_len=16,
    proprio_history=5,
    chunk_size=50,
    execute_steps=10,
    max_steps=300,
    normalize_embeddings=True,
    image_size=256,
):
    """Run episode and collect video frames"""
    obs = env.reset()

    # Buffers
    frame_buffer = deque(maxlen=video_len)
    proprio_buffer = deque(maxlen=proprio_history)
    action_buffer = []

    # Video recording
    video_frames = []

    total_reward = 0
    success = False

    for step in range(max_steps):
        # Get current observation
        current_image = obs['agentview_image']
        current_proprio = extract_proprio(obs)

        # Record frame for video
        video_frames.append(current_image.copy())

        # Update buffers
        frame_buffer.append(current_image)
        proprio_buffer.append(current_proprio)

        # Pad if needed
        while len(frame_buffer) < video_len:
            frame_buffer.appendleft(frame_buffer[0])
        while len(proprio_buffer) < proprio_history:
            proprio_buffer.appendleft(proprio_buffer[0])

        # Get action
        if len(action_buffer) == 0:
            # Prepare inputs
            video = np.stack(list(frame_buffer))
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
            video_tensor = video_tensor.unsqueeze(0).to(device)

            proprio = np.stack(list(proprio_buffer))
            proprio_tensor = torch.from_numpy(proprio).float().unsqueeze(0).to(device)

            # Encode video
            with torch.no_grad():
                current_emb = model.vjepa2.encode_video(video_tensor)
                if normalize_embeddings:
                    current_emb = F.normalize(current_emb, dim=-1)

                # Get actions
                action_chunk = model.forward_with_precomputed(current_emb, goal_emb, proprio_tensor)
                action_chunk = action_chunk[0].cpu().numpy()

            action_buffer = list(action_chunk[:execute_steps])

        # Execute action
        action = action_buffer.pop(0)
        action[6] = 1.0 if action[6] > 0 else -1.0  # Binary gripper

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            if info.get('success', False):
                success = True
            elif hasattr(env.env, '_check_success'):
                success = env.env._check_success()
            break

    # Check success at end
    if not success and hasattr(env.env, '_check_success'):
        success = env.env._check_success()

    return {
        'success': success,
        'total_reward': total_reward,
        'episode_length': step + 1,
        'video_frames': video_frames,
    }


def save_gif(frames, path, fps=20):
    """Save frames as GIF"""
    # Frames are (H, W, C) uint8
    duration_ms = 1000 / fps
    imageio.mimsave(path, frames, duration=duration_ms)


def main():
    parser = argparse.ArgumentParser(description='V7 Evaluation with Video Recording')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--suite', type=str, default='libero_spatial')
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='results/v7/videos')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    encoder_cfg = config.get('encoder', {})
    data_cfg = config.get('data', {})

    # Create model
    print("Creating model...")
    model = VJEPA2Policy(
        vjepa2_model_path=encoder_cfg.get('model_path', '/workspace/models/vjepa2-ac-vitg.pt'),
        vjepa2_model_name=encoder_cfg.get('model_name', 'vjepa2_vitg'),
        vjepa2_freeze=True,
        vjepa2_num_frames=encoder_cfg.get('num_frames', 16),
        proprio_dim=model_cfg.get('proprio_dim', 15),
        proprio_history=model_cfg.get('proprio_history', 5),
        embed_dim=model_cfg.get('embed_dim', 1408),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        num_heads=model_cfg.get('num_heads', 8),
        num_layers=model_cfg.get('num_layers', 4),
        num_spatial_tokens=model_cfg.get('num_spatial_tokens', 64),
        action_dim=model_cfg.get('action_dim', 7),
        chunk_size=model_cfg.get('chunk_size', 50),
        separate_gripper_head=model_cfg.get('separate_gripper_head', False),
        device=args.device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")

    # Get benchmark
    BenchClass = benchmark.get_benchmark(args.suite)
    bench = BenchClass()
    task_names = bench.get_task_names()
    n_tasks = len(task_names)

    print(f"\nEvaluating {n_tasks} tasks, {args.n_episodes} episodes each")
    print("=" * 60)

    results = {}
    overall_successes = 0
    overall_episodes = 0

    for task_idx, task_name in enumerate(task_names):
        print(f"\nTask {task_idx + 1}/{n_tasks}: {task_name}")

        # Get task info
        task = bench.get_task(task_idx)
        bddl_file = bench.get_task_bddl_file_path(task_idx)
        init_states = bench.get_task_init_states(task_idx)

        # Get goal image from demo HDF5 file
        import h5py
        demo_dir = Path("/workspace/data/libero") / task.problem_folder
        demo_file = demo_dir / f"{task.name}_demo.hdf5"

        goal_image = None
        if demo_file.exists():
            with h5py.File(demo_file, 'r') as f:
                for key in ['agentview_rgb', 'agentview_image', 'rgb']:
                    full_key = f'data/demo_0/obs/{key}'
                    if full_key in f:
                        goal_image = f[full_key][-1]
                        break

        if goal_image is None:
            # Fallback: use reset observation
            env_temp = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                render_camera="agentview",
                camera_heights=256,
                camera_widths=256,
            )
            obs = env_temp.reset()
            goal_image = obs['agentview_image']
            env_temp.close()
            print(f"  Warning: Using reset obs as goal for {task_name}")

        # Encode goal
        goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
        goal_tensor = goal_tensor.unsqueeze(0).unsqueeze(0).to(args.device)  # (1, 1, C, H, W)
        goal_tensor = goal_tensor.repeat(1, 16, 1, 1, 1)  # Repeat for video encoder

        with torch.no_grad():
            goal_emb = model.vjepa2.encode_video(goal_tensor)
            if data_cfg.get('normalize', True):
                goal_emb = F.normalize(goal_emb, dim=-1)

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            render_camera="agentview",
            camera_heights=256,
            camera_widths=256,
        )

        task_successes = 0

        for ep in range(args.n_episodes):
            print(f"  Episode {ep + 1}/{args.n_episodes}", end=" ")

            # Set initial state
            init_state = init_states[ep % len(init_states)]
            env.reset()
            env.set_init_state(init_state)

            # Run episode
            result = run_episode_with_video(
                model=model,
                env=env,
                goal_emb=goal_emb,
                device=args.device,
                video_len=encoder_cfg.get('num_frames', 16),
                proprio_history=model_cfg.get('proprio_history', 5),
                chunk_size=model_cfg.get('chunk_size', 50),
                execute_steps=10,
                max_steps=300,
                normalize_embeddings=data_cfg.get('normalize', True),
            )

            success_str = "SUCCESS" if result['success'] else "FAIL"
            print(f"- {success_str} (len={result['episode_length']}, reward={result['total_reward']:.2f})")

            if result['success']:
                task_successes += 1
                overall_successes += 1
            overall_episodes += 1

            # Save video
            video_path = output_dir / f"task{task_idx + 1:02d}_ep{ep + 1}_{success_str.lower()}.gif"
            save_gif(result['video_frames'], str(video_path), fps=20)
            print(f"    Saved: {video_path}")

        env.close()

        task_rate = task_successes / args.n_episodes
        results[task_name] = task_rate
        print(f"  Task success rate: {task_rate * 100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    overall_rate = overall_successes / overall_episodes
    print(f"\nOverall: {overall_rate * 100:.1f}% ({overall_successes}/{overall_episodes})")
    print("\nPer-task:")
    for task_name, rate in results.items():
        status = "✓" if rate > 0 else "✗"
        print(f"  {status} {task_name}: {rate * 100:.1f}%")

    # Save results
    results_file = output_dir / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"V7 Evaluation Results\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Overall: {overall_rate * 100:.1f}%\n\n")
        for task_name, rate in results.items():
            f.write(f"{task_name}: {rate * 100:.1f}%\n")
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
