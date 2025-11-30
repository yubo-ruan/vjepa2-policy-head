#!/usr/bin/env python3
"""
Record evaluation episodes as GIF videos.

Records robot execution with the trained policy and saves as animated GIFs.

Usage:
    python scripts/record_episodes.py --checkpoint /workspace/checkpoints_v3/best_model.pt \
        --suite libero_spatial --task 2 --n_episodes 3 --output_dir results/v3/videos
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.policy import VJEPA2Policy
from vjepa_policy.utils.evaluation import LIBEROEvaluatorSpatial, VALID_SUITES

try:
    from libero.libero import benchmark
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("Warning: LIBERO not installed")


class GIFRecorder:
    """Records episode frames and saves as GIF."""

    def __init__(self, output_path: str, fps: int = 15):
        self.output_path = Path(output_path)
        self.fps = fps
        self.frames = []

    def add_frame(self, frame: np.ndarray, action: np.ndarray = None, step: int = 0, success: bool = None):
        """Add a frame with optional action overlay."""
        # Ensure RGB format
        if frame.shape[-1] != 3:
            frame = frame.transpose(1, 2, 0)

        # Flip vertically - LIBERO images are upside down
        frame = np.flipud(frame).copy()

        # Create PIL image
        img = Image.fromarray(frame.astype(np.uint8))

        # Resize for better GIF quality
        img = img.resize((256, 256), Image.LANCZOS)

        # Add text overlay
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Step counter
        draw.text((5, 5), f"Step: {step}", fill=(255, 255, 255), font=font)

        # Action info
        if action is not None:
            grip_state = "OPEN" if action[6] > 0 else "CLOSE"
            draw.text((5, 20), f"Grip: {grip_state}", fill=(255, 255, 0), font=font)

        # Success indicator
        if success is not None:
            color = (0, 255, 0) if success else (255, 0, 0)
            text = "SUCCESS" if success else "FAIL"
            draw.text((5, 240), text, fill=color, font=font)

        self.frames.append(np.array(img))

    def save(self):
        """Save frames as GIF."""
        if not self.frames:
            print("No frames to save")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as GIF
        imageio.mimsave(
            str(self.output_path),
            self.frames,
            fps=self.fps,
            loop=0  # Loop forever
        )
        print(f"Saved GIF: {self.output_path} ({len(self.frames)} frames)")

    def reset(self):
        """Clear frames for new episode."""
        self.frames = []


class RecordingEvaluator(LIBEROEvaluatorSpatial):
    """Extended evaluator that records episodes as GIFs."""

    def __init__(self, *args, output_dir: str = "videos", **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run_episode_with_recording(
        self,
        env,
        goal_tokens: torch.Tensor,
        recorder: GIFRecorder,
        verbose: bool = False,
    ):
        """Run episode and record frames."""
        import torch.nn.functional as F
        from collections import defaultdict

        obs = env.reset()

        frame_buffer = []
        proprio_buffer = []
        all_actions = []
        action_history = defaultdict(list)

        total_reward = 0
        success = False

        for step in range(self.max_episode_steps):
            current_image = obs['agentview_image']
            current_proprio = self.extract_proprio(obs)

            frame_buffer.append(current_image)
            proprio_buffer.append(current_proprio)

            if len(frame_buffer) > self.video_len:
                frame_buffer.pop(0)
            if len(proprio_buffer) > self.proprio_history:
                proprio_buffer.pop(0)

            while len(frame_buffer) < self.video_len:
                frame_buffer.insert(0, frame_buffer[0])
            while len(proprio_buffer) < self.proprio_history:
                proprio_buffer.insert(0, proprio_buffer[0])

            need_replan = (step % self.execute_steps == 0) or (step not in action_history)

            if need_replan:
                video = np.stack(frame_buffer)
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
                video_tensor = video_tensor.unsqueeze(0).to(self.device)

                if video_tensor.shape[-1] != self.image_size or video_tensor.shape[-2] != self.image_size:
                    B, T, C, H, W = video_tensor.shape
                    video_tensor = video_tensor.view(B * T, C, H, W)
                    video_tensor = F.interpolate(
                        video_tensor, size=(self.image_size, self.image_size),
                        mode='bilinear', align_corners=False
                    )
                    video_tensor = video_tensor.view(B, T, C, self.image_size, self.image_size)

                proprio = np.stack(proprio_buffer)
                proprio_tensor = torch.from_numpy(proprio).float()
                proprio_tensor = proprio_tensor.unsqueeze(0).to(self.device)

                video_tokens = self.model.vjepa2.encode_video_spatial(video_tensor)

                if self.normalize_embeddings:
                    video_tokens = F.normalize(video_tokens, dim=-1)

                action_chunk = self.model.forward_with_precomputed(
                    video_tokens, goal_tokens, proprio_tensor
                )
                action_chunk = action_chunk[0].cpu().numpy()

                if self.use_temporal_ensemble:
                    for i in range(min(self.chunk_size, self.max_episode_steps - step)):
                        future_step = step + i
                        weight = np.exp(-0.1 * i)
                        action_history[future_step].append((action_chunk[i], weight))
                else:
                    for i in range(self.execute_steps):
                        future_step = step + i
                        if future_step < self.max_episode_steps:
                            action_history[future_step] = [(action_chunk[i], 1.0)]

            if self.use_temporal_ensemble and step in action_history and len(action_history[step]) > 0:
                predictions = action_history[step]
                weights = np.array([w for _, w in predictions])
                actions = np.array([a for a, _ in predictions])
                weights = weights / weights.sum()

                action = np.zeros(7)
                action[:6] = np.sum(actions[:, :6] * weights[:, None], axis=0)

                gripper_votes = [1 if a[6] > 0 else -1 for a, _ in predictions]
                weighted_votes = sum(v * w for v, w in zip(gripper_votes, weights))
                action[6] = 1.0 if weighted_votes > 0 else -1.0
            else:
                if step in action_history and len(action_history[step]) > 0:
                    action = action_history[step][0][0].copy()
                    action[6] = 1.0 if action[6] > 0 else -1.0
                else:
                    action = np.zeros(7)

            # Record frame (every 2 steps to reduce GIF size)
            if step % 2 == 0:
                recorder.add_frame(current_image, action, step)

            all_actions.append(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                if info.get('success', False):
                    success = True
                elif hasattr(env.env, '_check_success'):
                    success = env.env._check_success()
                elif reward > 0.9:
                    success = True
                break

            if step in action_history:
                del action_history[step]

        # Add final frame with result
        recorder.add_frame(obs['agentview_image'], action, step, success)

        return {
            'success': success,
            'total_reward': total_reward,
            'episode_length': step + 1,
            'actions': np.array(all_actions),
        }


def main():
    parser = argparse.ArgumentParser(description='Record evaluation episodes as GIFs')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=VALID_SUITES,
                        help='LIBERO suite')
    parser.add_argument('--task', type=int, default=None,
                        help='Task index to record (0-9). If not specified, records all tasks.')
    parser.add_argument('--all_tasks', action='store_true',
                        help='Record all 10 tasks (T1-T10)')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to record per task')
    parser.add_argument('--output_dir', type=str, default='results/v3/videos',
                        help='Output directory for GIFs')
    parser.add_argument('--fps', type=int, default=15,
                        help='GIF frames per second')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')

    args = parser.parse_args()

    # Determine which tasks to record
    if args.all_tasks or args.task is None:
        tasks_to_record = list(range(10))  # All 10 tasks (0-9)
    else:
        tasks_to_record = [args.task]

    if not LIBERO_AVAILABLE:
        print("Error: LIBERO not installed")
        return

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    encoder_cfg = config.get('encoder', {})
    data_cfg = config.get('data', {})

    # Create model
    use_goal_conditioned = model_cfg.get('use_goal_conditioned', False)
    print("Creating model...")
    if use_goal_conditioned:
        print("  Using GoalConditionedPolicyHead (goal-dependent action queries)")
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
        use_goal_conditioned=use_goal_conditioned,
        device=args.device,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")

    # Create evaluator
    evaluator = RecordingEvaluator(
        model=model,
        device=args.device,
        video_len=encoder_cfg.get('num_frames', 16),
        proprio_history=model_cfg.get('proprio_history', 5),
        chunk_size=model_cfg.get('chunk_size', 50),
        execute_steps=10,
        max_episode_steps=300,
        normalize_embeddings=data_cfg.get('normalize', True),
        n_spatial_tokens=model_cfg.get('num_spatial_tokens', 64),
        use_temporal_ensemble=True,
        output_dir=args.output_dir,
    )

    # Get benchmark
    bench_class = benchmark.get_benchmark(args.suite)
    bench = bench_class()

    # Import F for goal encoding
    import torch.nn.functional as F

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print overall plan
    print(f"\n{'='*60}")
    print(f"Recording Episodes for {len(tasks_to_record)} Tasks")
    print(f"{'='*60}")
    print(f"Tasks: {[t+1 for t in tasks_to_record]} (T1-T{max(tasks_to_record)+1})")
    print(f"Episodes per task: {args.n_episodes}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Track overall results
    all_task_results = {}

    # Loop through all tasks
    for task_idx in tasks_to_record:
        task = bench.get_task(task_idx)
        bddl_file = bench.get_task_bddl_file_path(task_idx)
        task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"

        print(f"\n{'='*50}")
        print(f"Task {task_idx + 1}/10 (T{task_idx + 1}): {task_name}")
        print(f"{'='*50}")

        # Create environment for this task
        env = evaluator.create_env(bddl_file)
        goal_image = evaluator.get_goal_image(env, task)

        # Encode goal
        goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
        goal_tensor = goal_tensor.unsqueeze(0).to(args.device)
        if goal_tensor.shape[-1] != 256:
            goal_tensor = F.interpolate(goal_tensor, size=(256, 256), mode='bilinear', align_corners=False)

        goal_tokens = model.vjepa2.encode_image_spatial(goal_tensor)
        if data_cfg.get('normalize', True):
            goal_tokens = F.normalize(goal_tokens, dim=-1)

        # Record episodes for this task
        task_successes = []
        for ep in range(args.n_episodes):
            print(f"  Episode {ep + 1}/{args.n_episodes}...", end=" ")

            # Create recorder
            safe_name = task_name.replace(' ', '_').replace('/', '_')[:50]
            gif_path = output_dir / f"T{task_idx+1}_task{task_idx}_{safe_name}_ep{ep+1}.gif"
            recorder = GIFRecorder(gif_path, fps=args.fps)

            # Run episode
            result = evaluator.run_episode_with_recording(env, goal_tokens, recorder, verbose=False)

            # Save GIF
            recorder.save()

            status = "SUCCESS" if result['success'] else "FAIL"
            print(f"{status} - Length: {result['episode_length']}, Reward: {result['total_reward']:.2f}")
            task_successes.append(result['success'])

        env.close()

        # Task summary
        success_rate = 100 * sum(task_successes) / len(task_successes)
        all_task_results[task_idx] = {
            'name': task_name,
            'successes': task_successes,
            'success_rate': success_rate
        }
        print(f"  Task {task_idx + 1} Success Rate: {sum(task_successes)}/{len(task_successes)} ({success_rate:.0f}%)")

    # Overall summary
    print(f"\n{'='*60}")
    print(f"Recording Complete - All Tasks Summary")
    print(f"{'='*60}")

    total_successes = 0
    total_episodes = 0
    for task_idx in sorted(all_task_results.keys()):
        result = all_task_results[task_idx]
        successes = sum(result['successes'])
        episodes = len(result['successes'])
        total_successes += successes
        total_episodes += episodes
        print(f"T{task_idx+1}: {result['name'][:40]:40s} - {successes}/{episodes} ({result['success_rate']:.0f}%)")

    overall_rate = 100 * total_successes / total_episodes if total_episodes > 0 else 0
    print(f"\n{'='*60}")
    print(f"Overall Success Rate: {total_successes}/{total_episodes} ({overall_rate:.1f}%)")
    print(f"Videos saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
