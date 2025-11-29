"""
Visualize policy behavior by recording evaluation episodes.

Saves annotated videos showing:
- Robot's camera view
- Predicted actions overlay (position, rotation, gripper)
- Step counter and success status

Usage:
    python scripts/visualize_policy.py --checkpoint /workspace/checkpoints_spatial/best_model.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import yaml
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2PolicySpatial
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


class VideoBuffer:
    """Buffer for maintaining video frame history."""

    def __init__(self, size: int = 16, image_size: int = 256):
        self.size = size
        self.image_size = image_size
        self.frames = []

    def add(self, frame: np.ndarray):
        """Add a frame to the buffer."""
        self.frames.append(frame.copy())
        if len(self.frames) > self.size:
            self.frames.pop(0)

    def get_tensor(self, device: str = "cuda") -> torch.Tensor:
        """Get frames as tensor (1, T, C, H, W)."""
        # Pad with first frame if not enough frames
        while len(self.frames) < self.size:
            self.frames.insert(0, self.frames[0].copy())

        # Stack and convert
        frames = np.stack(self.frames[-self.size:], axis=0)  # (T, H, W, C)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

        # Resize if needed
        if tensor.shape[-1] != self.image_size:
            tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode='bilinear')

        return tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

    def reset(self):
        """Clear the buffer."""
        self.frames = []


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load trained spatial model."""
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


def get_proprio(obs, proprio_dim: int = 15) -> np.ndarray:
    """Extract proprioception from observation."""
    proprio_parts = []

    # Try different proprio keys
    for key in ['ee_pos', 'ee_states', 'gripper_states', 'joint_states']:
        if key in obs:
            proprio_parts.append(obs[key].flatten())

    if proprio_parts:
        proprio = np.concatenate(proprio_parts)[:proprio_dim]
    else:
        proprio = np.zeros(proprio_dim)

    return proprio.astype(np.float32)


def annotate_frame(frame: np.ndarray, action: np.ndarray, step: int,
                   gt_action: np.ndarray = None, success: bool = False) -> np.ndarray:
    """Add action information overlay to frame."""
    frame = frame.copy()

    # Get dimensions
    h, w = frame.shape[:2]

    # Add black bar at top for text
    overlay_height = 100
    overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
    frame = np.vstack([overlay, frame])

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    cyan = (255, 255, 0)

    # Step counter
    cv2.putText(frame, f"Step: {step}", (10, 18), font, 0.5, white, 1)

    # Success indicator
    if success:
        cv2.putText(frame, "SUCCESS!", (w - 100, 18), font, 0.6, green, 2)

    # Predicted action - Position
    cv2.putText(frame, f"Pred pos: dx:{action[0]:+.3f} dy:{action[1]:+.3f} dz:{action[2]:+.3f}",
                (10, 38), font, 0.4, green, 1)

    # Predicted action - Rotation
    cv2.putText(frame, f"Pred rot: rx:{action[3]:+.3f} ry:{action[4]:+.3f} rz:{action[5]:+.3f}",
                (10, 53), font, 0.4, yellow, 1)

    # Gripper
    gripper_val = action[6]
    gripper_text = "CLOSE" if gripper_val > 0 else "OPEN"
    gripper_color = red if gripper_val > 0 else green
    cv2.putText(frame, f"Gripper: {gripper_text} ({gripper_val:+.2f})",
                (10, 68), font, 0.4, gripper_color, 1)

    # Ground truth action (if available)
    if gt_action is not None:
        cv2.putText(frame, f"GT pos:   dx:{gt_action[0]:+.3f} dy:{gt_action[1]:+.3f} dz:{gt_action[2]:+.3f}",
                    (10, 83), font, 0.4, cyan, 1)

        # Position error
        pos_error = np.abs(action[:3] - gt_action[:3]).mean()
        cv2.putText(frame, f"Pos Err: {pos_error:.3f}", (w - 120, 38), font, 0.4,
                    red if pos_error > 0.15 else green, 1)

    # Action magnitude bar (visual indicator)
    action_mag = np.linalg.norm(action[:3])
    bar_width = int(min(action_mag * 500, w - 20))
    cv2.rectangle(frame, (10, 95), (10 + bar_width, 98), green, -1)

    return frame


def save_video(frames: list, path: str, fps: int = 30):
    """Save frames as MP4 video."""
    if len(frames) == 0:
        print("No frames to save!")
        return

    h, w = frames[0].shape[:2]

    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for frame in frames:
        # Ensure BGR format for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)

    out.release()
    print(f"Saved {len(frames)} frames to {path}")


def save_gif(frames: list, path: str, fps: int = 10, flip: bool = True):
    """Save frames as GIF (viewable in VSCode).

    Args:
        frames: List of RGB frames
        path: Output path (will be saved as .gif)
        fps: Frames per second (lower = smaller file)
        flip: Whether to vertically flip (fixes upside-down simulation render)
    """
    if len(frames) == 0:
        print("No frames to save!")
        return

    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, skipping GIF save")
        return

    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        # Flip vertically if needed (simulation renders upside-down)
        if flip:
            frame = np.flipud(frame)

        # Ensure RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            img = Image.fromarray(frame.astype(np.uint8), mode='RGB')
        else:
            img = Image.fromarray(frame.astype(np.uint8))

        # Resize for smaller file
        img = img.resize((256, int(img.height * 256 / img.width)), Image.LANCZOS)
        pil_frames.append(img)

    # Sample frames to reduce size (every nth frame)
    sample_rate = max(1, len(pil_frames) // 100)  # Keep ~100 frames max
    sampled_frames = pil_frames[::sample_rate]

    # Save as GIF
    gif_path = path.replace('.mp4', '.gif')
    sampled_frames[0].save(
        gif_path,
        save_all=True,
        append_images=sampled_frames[1:],
        duration=int(1000 / fps),  # ms per frame
        loop=0
    )
    print(f"Saved GIF ({len(sampled_frames)} frames) to {gif_path}")


def evaluate_and_record(
    model,
    config: dict,
    task,
    bddl_file_path: str,
    save_dir: str,
    task_idx: int = 0,
    episode_idx: int = 0,
    max_steps: int = 300,
    device: str = "cuda",
):
    """Evaluate one episode and record video."""

    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        render_camera="agentview",
        camera_heights=256,
        camera_widths=256,
    )

    # Reset
    obs = env.reset()

    # Get goal image (from demo)
    import h5py
    demo_dir = Path("/workspace/data/libero") / task.problem_folder
    demo_file = demo_dir / f"{task.name}_demo.hdf5"

    with h5py.File(demo_file, 'r') as f:
        # Get goal from last frame of first demo
        for key in ['agentview_rgb', 'agentview_image']:
            full_key = f'data/demo_0/obs/{key}'
            if full_key in f:
                goal_image = f[full_key][-1]
                break
        else:
            goal_image = obs['agentview_image']

    # Prepare goal tensor
    goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
    goal_tensor = goal_tensor.unsqueeze(0).to(device)
    if goal_tensor.shape[-1] != 256:
        goal_tensor = F.interpolate(goal_tensor, size=(256, 256), mode='bilinear')

    # Encode goal once
    with torch.no_grad():
        goal_tokens = model.vjepa2.encode_image_spatial(goal_tensor)
        goal_tokens = F.normalize(goal_tokens, dim=-1)

    # Initialize buffers
    video_buffer = VideoBuffer(size=config['vjepa2']['num_frames'], image_size=256)
    proprio_history = []
    proprio_dim = config['proprio']['dim']
    proprio_len = config['proprio']['history_len']

    # Recording
    frames = []
    actions_taken = []
    success = False

    # Get initial frame
    current_frame = obs['agentview_image']
    video_buffer.add(current_frame)

    print(f"Recording episode for task {task_idx}, episode {episode_idx}...")

    for step in range(max_steps):
        # Get proprio
        proprio = get_proprio(obs, proprio_dim)
        proprio_history.append(proprio)
        if len(proprio_history) > proprio_len:
            proprio_history.pop(0)

        # Pad proprio history
        while len(proprio_history) < proprio_len:
            proprio_history.insert(0, proprio_history[0].copy())

        proprio_tensor = torch.from_numpy(np.stack(proprio_history[-proprio_len:])).float()
        proprio_tensor = proprio_tensor.unsqueeze(0).to(device)

        # Get video tensor
        video_tensor = video_buffer.get_tensor(device)

        # Predict action
        with torch.no_grad():
            video_tokens = model.vjepa2.encode_video_spatial(video_tensor)
            video_tokens = F.normalize(video_tokens, dim=-1)
            pred_actions = model.forward_with_precomputed(video_tokens, goal_tokens, proprio_tensor)
            action = pred_actions[0, 0].cpu().numpy()

        # Threshold gripper
        action[6] = 1.0 if action[6] > 0 else -1.0

        # Annotate and store frame
        annotated = annotate_frame(current_frame, action, step, success=success)
        frames.append(annotated)
        actions_taken.append(action.copy())

        # Execute action
        obs, reward, done, info = env.step(action)

        # Update frame buffer
        current_frame = obs['agentview_image']
        video_buffer.add(current_frame)

        # Check success
        if hasattr(env.env, '_check_success'):
            success = env.env._check_success()
        elif reward > 0.5:
            success = True

        if success:
            print(f"  SUCCESS at step {step}!")
            # Add a few more frames with success overlay
            for _ in range(30):
                annotated = annotate_frame(current_frame, action, step, success=True)
                frames.append(annotated)
            break

        if done:
            break

    env.close()

    # Save video and GIF
    task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"
    task_name_clean = task_name[:50].replace(" ", "_").replace("/", "_")
    video_path = f"{save_dir}/task{task_idx}_ep{episode_idx}_{task_name_clean}.mp4"
    save_video(frames, video_path, fps=30)

    # Save GIF (viewable in VSCode)
    save_gif(frames, video_path, fps=10, flip=True)

    # Save actions
    actions_path = video_path.replace('.mp4', '_actions.npy')
    np.save(actions_path, np.array(actions_taken))
    print(f"  Saved actions to {actions_path}")

    return {
        'success': success,
        'steps': len(actions_taken),
        'video_path': video_path,
        'gif_path': video_path.replace('.mp4', '.gif'),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize policy behavior")
    parser.add_argument('--checkpoint', type=str,
                        default='/workspace/checkpoints_spatial/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/spatial.yaml',
                        help='Path to config file')
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        help='LIBERO suite')
    parser.add_argument('--task_idx', type=int, default=0,
                        help='Task index to visualize')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to record')
    parser.add_argument('--save_dir', type=str, default='/workspace/videos',
                        help='Directory to save videos')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Maximum steps per episode')
    args = parser.parse_args()

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.config, device)

    # Get benchmark
    BenchClass = benchmark.get_benchmark(args.suite)
    bench = BenchClass()

    # Get task
    task = bench.get_task(args.task_idx)
    bddl_file_path = bench.get_task_bddl_file_path(args.task_idx)
    task_name = task.language if hasattr(task, 'language') else f"task_{args.task_idx}"

    print(f"\nTask {args.task_idx}: {task_name}")
    print(f"Recording {args.n_episodes} episodes...")
    print(f"Saving to: {args.save_dir}")
    print("=" * 60)

    results = []
    for ep in range(args.n_episodes):
        result = evaluate_and_record(
            model=model,
            config=config,
            task=task,
            bddl_file_path=bddl_file_path,
            save_dir=args.save_dir,
            task_idx=args.task_idx,
            episode_idx=ep,
            max_steps=args.max_steps,
            device=device,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successes = sum(r['success'] for r in results)
    print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")
    print(f"\nVideos saved to: {args.save_dir}/")
    for r in results:
        status = "SUCCESS" if r['success'] else "FAIL"
        print(f"  {Path(r['video_path']).name}: {status} ({r['steps']} steps)")


if __name__ == "__main__":
    main()
