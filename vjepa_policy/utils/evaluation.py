"""
LIBERO Benchmark Evaluation

Evaluates trained policy on LIBERO benchmark suites.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

# LIBERO imports
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    print("Warning: LIBERO not installed. Install with: pip install libero")
    LIBERO_AVAILABLE = False


# Valid LIBERO suites (libero_long does NOT exist)
VALID_SUITES = [
    'libero_object',
    'libero_spatial',
    'libero_goal',
    'libero_90',
    'libero_10',
]


class LIBEROEvaluator:
    """
    Evaluator for LIBERO benchmark.

    Supports LIBERO suites:
    - libero_object: Object manipulation (10 tasks)
    - libero_spatial: Spatial reasoning (10 tasks)
    - libero_goal: Goal-conditioned (10 tasks)
    - libero_90: 90 tasks from LIBERO-100
    - libero_10: 10 tasks from LIBERO-100
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        video_len: int = 16,
        proprio_history: int = 5,
        chunk_size: int = 50,
        execute_steps: int = 10,
        max_episode_steps: int = 300,
        image_size: int = 256,
        normalize_embeddings: bool = True,
    ):
        """
        Args:
            model: Trained VJEPA2Policy model
            device: Device to run on
            video_len: Number of frames for video input
            proprio_history: Number of proprio history steps
            chunk_size: Action chunk size from policy
            execute_steps: How many actions to execute before replanning
            max_episode_steps: Maximum steps per episode
            image_size: Image resolution
            normalize_embeddings: Whether to L2-normalize embeddings (should match training)
        """
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO not available. Install with: pip install libero")

        self.model = model
        self.device = device
        self.video_len = video_len
        self.proprio_history = proprio_history
        self.chunk_size = chunk_size
        self.execute_steps = execute_steps
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self.normalize_embeddings = normalize_embeddings

        self.model.eval()

    def get_benchmark(self, suite_name: str):
        """Get LIBERO benchmark suite"""
        if suite_name not in VALID_SUITES:
            raise ValueError(f"Invalid suite: {suite_name}. Valid: {VALID_SUITES}")
        # get_benchmark returns a class, we need to instantiate it
        BenchClass = benchmark.get_benchmark(suite_name)
        return BenchClass()

    def create_env(self, bddl_file_path: str):
        """Create environment for a task"""
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            render_camera="agentview",
            camera_heights=self.image_size,
            camera_widths=self.image_size,
        )
        return env

    def get_goal_image(self, env, task, demo_dir: str = "/workspace/data/libero") -> np.ndarray:
        """
        Get goal image for the task.

        For LIBERO, we use the final frame from a successful demo.
        """
        import h5py

        # Find demo file
        suite_dir = Path(demo_dir) / task.problem_folder
        demo_file = suite_dir / f"{task.name}_demo.hdf5"

        if demo_file.exists():
            with h5py.File(demo_file, 'r') as f:
                # Get last frame from first demo
                # Try different possible image keys
                for key in ['agentview_rgb', 'agentview_image', 'rgb']:
                    full_key = f'data/demo_0/obs/{key}'
                    if full_key in f:
                        return f[full_key][-1]

        # Fallback: Use current observation as placeholder
        print(f"Warning: Demo file not found: {demo_file}")
        print("Using current observation as goal (not ideal)")
        obs = env.reset()
        return obs['agentview_image']

    def extract_proprio(self, obs: Dict) -> np.ndarray:
        """Extract proprioception from observation dict

        Must match training format: ee_pos(3) + ee_ori(3) + gripper(2) + joint_pos(7) = 15 dims
        Live env provides quat (4D), so we convert to euler angles (3D).
        """
        proprio_parts = []

        # End-effector position (3D)
        if 'robot0_eef_pos' in obs:
            proprio_parts.append(obs['robot0_eef_pos'])

        # End-effector orientation: convert quaternion (4D) to euler (3D)
        if 'robot0_eef_quat' in obs:
            quat = obs['robot0_eef_quat']
            euler = self._quat_to_euler(quat)
            proprio_parts.append(euler)

        # Gripper state (2D)
        if 'robot0_gripper_qpos' in obs:
            proprio_parts.append(obs['robot0_gripper_qpos'])

        # Joint positions (7D) - NO joint velocities!
        if 'robot0_joint_pos' in obs:
            proprio_parts.append(obs['robot0_joint_pos'])

        if not proprio_parts:
            raise ValueError(f"No proprio found in obs. Keys: {obs.keys()}")

        result = np.concatenate(proprio_parts)
        # Verify we get 15 dims as expected
        if result.shape[0] != 15:
            print(f"Warning: Expected 15 proprio dims, got {result.shape[0]}")
        return result

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw]"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @torch.no_grad()
    def run_episode(
        self,
        env,
        goal_emb: torch.Tensor,
        verbose: bool = False,
    ) -> Dict:
        """
        Run a single evaluation episode.

        Args:
            env: LIBERO environment
            goal_emb: Pre-computed goal embedding (B=1, D)
            verbose: Print progress

        Returns:
            dict with 'success', 'total_reward', 'episode_length', 'actions'
        """
        obs = env.reset()

        # Initialize buffers
        frame_buffer = []  # For video encoding
        proprio_buffer = []  # For proprio history
        action_buffer = []  # Remaining actions from last chunk
        all_actions = []  # All executed actions

        total_reward = 0
        success = False

        for step in range(self.max_episode_steps):
            # Get current observation
            current_image = obs['agentview_image']
            current_proprio = self.extract_proprio(obs)

            # Update buffers
            frame_buffer.append(current_image)
            proprio_buffer.append(current_proprio)

            # Trim buffers to required length
            if len(frame_buffer) > self.video_len:
                frame_buffer.pop(0)
            if len(proprio_buffer) > self.proprio_history:
                proprio_buffer.pop(0)

            # Pad if not enough history
            while len(frame_buffer) < self.video_len:
                frame_buffer.insert(0, frame_buffer[0])
            while len(proprio_buffer) < self.proprio_history:
                proprio_buffer.insert(0, proprio_buffer[0])

            # Get action (replan if buffer empty)
            if len(action_buffer) == 0:
                # Prepare inputs
                video = np.stack(frame_buffer)  # (T, H, W, C)
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
                video_tensor = video_tensor.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

                # Resize video frames to model's expected size if needed
                if video_tensor.shape[-1] != self.image_size or video_tensor.shape[-2] != self.image_size:
                    B, T, C, H, W = video_tensor.shape
                    video_tensor = video_tensor.view(B * T, C, H, W)
                    video_tensor = F.interpolate(
                        video_tensor, size=(self.image_size, self.image_size),
                        mode='bilinear', align_corners=False
                    )
                    video_tensor = video_tensor.view(B, T, C, self.image_size, self.image_size)

                proprio = np.stack(proprio_buffer)  # (history, proprio_dim)
                proprio_tensor = torch.from_numpy(proprio).float()
                proprio_tensor = proprio_tensor.unsqueeze(0).to(self.device)  # (1, history, dim)

                # Encode current video
                current_emb = self.model.vjepa2.encode_video(video_tensor)

                # Normalize current embedding to match training
                if self.normalize_embeddings:
                    current_emb = F.normalize(current_emb, dim=-1)

                # Debug: print embedding and proprio info on first replan
                if step == 0 and verbose:
                    print(f"    Video emb norm: {current_emb.norm().item():.4f}")
                    print(f"    Goal emb norm: {goal_emb.norm().item():.4f}")
                    print(f"    Cos sim (current, goal): {F.cosine_similarity(current_emb, goal_emb, dim=-1).item():.4f}")
                    print(f"    Proprio[0]: {proprio[0, :6]}")

                # Get action chunk using precomputed goal embedding
                action_chunk = self.model.forward_with_precomputed(
                    current_emb, goal_emb, proprio_tensor
                )
                action_chunk = action_chunk[0].cpu().numpy()  # (chunk_size, action_dim)

                # Fill action buffer (take execute_steps actions)
                action_buffer = list(action_chunk[:self.execute_steps])

            # Execute action
            action = action_buffer.pop(0)

            # FIX: Threshold gripper to binary (-1 or 1)
            # This addresses high gripper error (~0.29) from model predictions
            action[6] = 1.0 if action[6] > 0 else -1.0

            all_actions.append(action)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if verbose and step % 50 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
                print(f"    Action: {action[:3]} (pos) {action[3:6]} (rot) {action[6]:.2f} (grip)")

            # Check success - LIBERO doesn't populate info['success'], use _check_success()
            if done:
                # First try info dict (some wrappers add it)
                if info.get('success', False):
                    success = True
                # Then try direct method call (LIBERO native)
                elif hasattr(env.env, '_check_success'):
                    success = env.env._check_success()
                # Finally check if reward indicates success
                elif reward > 0.9:
                    success = True
                break

        return {
            'success': success,
            'total_reward': total_reward,
            'episode_length': step + 1,
            'actions': np.array(all_actions),
        }

    def evaluate_task(
        self,
        task,
        bddl_file_path: str,
        n_episodes: int = 20,
        verbose: bool = False,
    ) -> Dict:
        """
        Evaluate on a single task.

        Returns:
            dict with success_rate, avg_reward, avg_length, per_episode results
        """
        env = self.create_env(bddl_file_path)
        goal_image = self.get_goal_image(env, task)

        # Pre-compute goal embedding ONCE for all episodes
        goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
        goal_tensor = goal_tensor.unsqueeze(0).to(self.device)

        # Resize goal image to model's expected size if needed (LIBERO demos are 128x128, model expects 256x256)
        if goal_tensor.shape[-1] != self.image_size or goal_tensor.shape[-2] != self.image_size:
            goal_tensor = F.interpolate(
                goal_tensor, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

        goal_emb = self.model.vjepa2.encode_image(goal_tensor)

        # Normalize goal embedding to match training
        if self.normalize_embeddings:
            goal_emb = F.normalize(goal_emb, dim=-1)

        results = []

        for ep in range(n_episodes):
            if verbose:
                print(f"  Episode {ep + 1}/{n_episodes}")

            result = self.run_episode(env, goal_emb, verbose=verbose)
            results.append(result)

            if verbose:
                status = "SUCCESS" if result['success'] else "FAIL"
                print(f"    {status} Length: {result['episode_length']}, "
                      f"Reward: {result['total_reward']:.2f}")

        env.close()

        # Aggregate results
        successes = [r['success'] for r in results]
        rewards = [r['total_reward'] for r in results]
        lengths = [r['episode_length'] for r in results]

        return {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'std_reward': np.std(rewards),
            'n_episodes': n_episodes,
            'per_episode': results,
        }

    def evaluate_suite(
        self,
        suite_name: str,
        n_episodes_per_task: int = 20,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate on entire benchmark suite.

        Args:
            suite_name: One of libero_object, libero_spatial, libero_goal, libero_90, libero_10
            n_episodes_per_task: Number of episodes per task
            verbose: Print progress

        Returns:
            dict with per-task results and aggregate metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {suite_name}")
        print(f"{'='*60}")

        bench = self.get_benchmark(suite_name)
        n_tasks = bench.get_num_tasks()

        all_results = {}
        all_successes = []

        for task_idx in range(n_tasks):
            task = bench.get_task(task_idx)
            bddl_file_path = bench.get_task_bddl_file_path(task_idx)
            task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"

            if verbose:
                print(f"\nTask {task_idx + 1}/{n_tasks}: {task_name}")

            result = self.evaluate_task(
                task,
                bddl_file_path=bddl_file_path,
                n_episodes=n_episodes_per_task,
                verbose=verbose,
            )

            all_results[task_name] = result
            all_successes.extend([r['success'] for r in result['per_episode']])

            if verbose:
                print(f"  Success rate: {result['success_rate']*100:.1f}%")

        # Aggregate across all tasks
        aggregate = {
            'suite_name': suite_name,
            'n_tasks': n_tasks,
            'n_episodes_per_task': n_episodes_per_task,
            'overall_success_rate': np.mean(all_successes),
            'per_task_success_rates': {
                name: r['success_rate'] for name, r in all_results.items()
            },
            'per_task_results': all_results,
        }

        print(f"\n{'='*60}")
        print(f"Suite Results: {suite_name}")
        print(f"{'='*60}")
        print(f"Overall Success Rate: {aggregate['overall_success_rate']*100:.1f}%")
        print(f"\nPer-task success rates:")
        for name, rate in aggregate['per_task_success_rates'].items():
            print(f"  {name}: {rate*100:.1f}%")

        return aggregate

    def evaluate_all_suites(
        self,
        suites: Optional[List[str]] = None,
        n_episodes_per_task: int = 20,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate on multiple LIBERO suites.

        Args:
            suites: List of suite names (default: object, spatial, goal)
            n_episodes_per_task: Episodes per task
            save_path: Path to save results JSON

        Returns:
            dict with results for all suites
        """
        if suites is None:
            # Default to the three 10-task suites
            suites = ['libero_object', 'libero_spatial', 'libero_goal']

        all_results = {}

        for suite in suites:
            if suite not in VALID_SUITES:
                print(f"Warning: Skipping invalid suite '{suite}'")
                continue

            result = self.evaluate_suite(
                suite,
                n_episodes_per_task=n_episodes_per_task,
                verbose=True,
            )
            all_results[suite] = result

        # Summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")

        for suite, result in all_results.items():
            print(f"{suite}: {result['overall_success_rate']*100:.1f}%")

        if len(all_results) > 0:
            avg_success = np.mean([r['overall_success_rate'] for r in all_results.values()])
            print(f"\nAverage across suites: {avg_success*100:.1f}%")

        # Save results
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            with open(save_path, 'w') as f:
                json.dump(convert_numpy(all_results), f, indent=2)

            print(f"\nResults saved to: {save_path}")

        return all_results


class LIBEROEvaluatorSpatial(LIBEROEvaluator):
    """
    Evaluator for spatial token models.

    Uses encode_video_spatial() and encode_image_spatial() instead of
    mean-pooled embeddings.
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        video_len: int = 16,
        proprio_history: int = 5,
        chunk_size: int = 50,
        execute_steps: int = 10,
        max_episode_steps: int = 300,
        image_size: int = 256,
        normalize_embeddings: bool = True,
        n_spatial_tokens: int = 64,
    ):
        # Don't call parent __init__ fully, just set attributes
        if not LIBERO_AVAILABLE:
            raise ImportError("LIBERO not available. Install with: pip install libero")

        self.model = model
        self.device = device
        self.video_len = video_len
        self.proprio_history = proprio_history
        self.chunk_size = chunk_size
        self.execute_steps = execute_steps
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self.normalize_embeddings = normalize_embeddings
        self.n_spatial_tokens = n_spatial_tokens

        self.model.eval()

    @torch.no_grad()
    def run_episode(
        self,
        env,
        goal_tokens: torch.Tensor,
        verbose: bool = False,
    ) -> Dict:
        """
        Run a single evaluation episode using spatial tokens.

        Args:
            env: LIBERO environment
            goal_tokens: Pre-computed goal spatial tokens (B=1, 64, D)
            verbose: Print progress

        Returns:
            dict with 'success', 'total_reward', 'episode_length', 'actions'
        """
        obs = env.reset()

        # Initialize buffers
        frame_buffer = []
        proprio_buffer = []
        action_buffer = []
        all_actions = []

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

            if len(action_buffer) == 0:
                # Prepare inputs
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

                # Encode using SPATIAL method: (B, 64, 1408)
                video_tokens = self.model.vjepa2.encode_video_spatial(video_tensor)

                # Normalize each token if configured
                if self.normalize_embeddings:
                    video_tokens = F.normalize(video_tokens, dim=-1)

                if step == 0 and verbose:
                    print(f"    Video tokens shape: {video_tokens.shape}")
                    print(f"    Goal tokens shape: {goal_tokens.shape}")
                    print(f"    Video tokens[0] norm: {video_tokens[0, 0].norm().item():.4f}")

                # Get action chunk using spatial tokens
                action_chunk = self.model.forward_with_precomputed(
                    video_tokens, goal_tokens, proprio_tensor
                )
                action_chunk = action_chunk[0].cpu().numpy()

                action_buffer = list(action_chunk[:self.execute_steps])

            action = action_buffer.pop(0)

            # FIX: Threshold gripper to binary (-1 or 1)
            # This addresses high gripper error (~0.29) from model predictions
            action[6] = 1.0 if action[6] > 0 else -1.0

            all_actions.append(action)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if verbose and step % 50 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")

            # Check success - LIBERO doesn't populate info['success'], use _check_success()
            if done:
                # First try info dict (some wrappers add it)
                if info.get('success', False):
                    success = True
                # Then try direct method call (LIBERO native)
                elif hasattr(env.env, '_check_success'):
                    success = env.env._check_success()
                # Finally check if reward indicates success
                elif reward > 0.9:
                    success = True
                break

        return {
            'success': success,
            'total_reward': total_reward,
            'episode_length': step + 1,
            'actions': np.array(all_actions),
        }

    def evaluate_task(
        self,
        task,
        bddl_file_path: str,
        n_episodes: int = 20,
        verbose: bool = False,
    ) -> Dict:
        """Evaluate on a single task using spatial tokens"""
        env = self.create_env(bddl_file_path)
        goal_image = self.get_goal_image(env, task)

        goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
        goal_tensor = goal_tensor.unsqueeze(0).to(self.device)

        if goal_tensor.shape[-1] != self.image_size or goal_tensor.shape[-2] != self.image_size:
            goal_tensor = F.interpolate(
                goal_tensor, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )

        # Encode goal using SPATIAL method: (B, 64, 1408)
        goal_tokens = self.model.vjepa2.encode_image_spatial(goal_tensor)

        if self.normalize_embeddings:
            goal_tokens = F.normalize(goal_tokens, dim=-1)

        results = []

        for ep in range(n_episodes):
            if verbose:
                print(f"  Episode {ep + 1}/{n_episodes}")

            result = self.run_episode(env, goal_tokens, verbose=verbose)
            results.append(result)

            if verbose:
                status = "SUCCESS" if result['success'] else "FAIL"
                print(f"    {status} Length: {result['episode_length']}, "
                      f"Reward: {result['total_reward']:.2f}")

        env.close()

        successes = [r['success'] for r in results]
        rewards = [r['total_reward'] for r in results]
        lengths = [r['episode_length'] for r in results]

        return {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'std_reward': np.std(rewards),
            'n_episodes': n_episodes,
            'per_episode': results,
        }


class DummyEvaluator:
    """
    Dummy evaluator for testing without LIBERO installed.
    """

    def __init__(self, model, **kwargs):
        self.model = model
        print("Using DummyEvaluator (LIBERO not installed)")

    def evaluate_suite(self, suite_name: str, **kwargs) -> Dict:
        """Return dummy results"""
        print(f"Dummy evaluation for {suite_name}")

        return {
            'suite_name': suite_name,
            'overall_success_rate': 0.0,
            'message': 'LIBERO not installed, returning dummy results',
        }

    def evaluate_all_suites(self, suites: Optional[List[str]] = None, **kwargs) -> Dict:
        """Return dummy results for all suites"""
        if suites is None:
            suites = ['libero_object', 'libero_spatial', 'libero_goal']
        return {suite: self.evaluate_suite(suite) for suite in suites}


def create_evaluator(model, **kwargs):
    """Factory function to create appropriate evaluator"""
    if LIBERO_AVAILABLE:
        return LIBEROEvaluator(model, **kwargs)
    else:
        return DummyEvaluator(model, **kwargs)
