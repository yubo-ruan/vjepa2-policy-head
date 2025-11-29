"""
LIBERO Benchmark Evaluation

Evaluates trained policy on LIBERO benchmark suites.
"""

import torch
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


class LIBEROEvaluator:
    """
    Evaluator for LIBERO benchmark.
    
    Supports all four LIBERO suites:
    - libero_object: Object manipulation (10 tasks)
    - libero_spatial: Spatial reasoning (10 tasks)  
    - libero_goal: Goal-conditioned (10 tasks)
    - libero_long: Long-horizon (10 tasks)
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
        
        self.model.eval()
    
    def get_benchmark(self, suite_name: str):
        """Get LIBERO benchmark suite"""
        return benchmark.get_benchmark(suite_name)
    
    def create_env(self, task):
        """Create environment for a task"""
        env = OffScreenRenderEnv(
            task.problem,
            task.language,
            render_camera="agentview",
            image_size=self.image_size,
        )
        return env
    
    def get_goal_image(self, env, task) -> np.ndarray:
        """
        Get goal image for the task.
        
        For LIBERO, we use the final frame from a successful demo
        or render the goal state if available.
        """
        # Option 1: Use goal from task definition if available
        if hasattr(task, 'goal_image'):
            return task.goal_image
        
        # Option 2: Load from demo file
        demo_path = task.demo_path if hasattr(task, 'demo_path') else None
        if demo_path and Path(demo_path).exists():
            import h5py
            with h5py.File(demo_path, 'r') as f:
                # Get last frame from first demo
                goal = f['data/demo_0/obs/agentview_image'][-1]
                return goal
        
        # Option 3: Use current observation as placeholder
        # (Not ideal, but allows testing)
        print("Warning: Using current observation as goal (no goal image found)")
        obs = env.get_observation()
        return obs['agentview_image']
    
    def extract_proprio(self, obs: Dict) -> np.ndarray:
        """Extract proprioception from observation dict"""
        proprio_parts = []
        
        # End-effector position
        if 'robot0_eef_pos' in obs:
            proprio_parts.append(obs['robot0_eef_pos'])
        
        # End-effector quaternion
        if 'robot0_eef_quat' in obs:
            proprio_parts.append(obs['robot0_eef_quat'])
        
        # Gripper state
        if 'robot0_gripper_qpos' in obs:
            proprio_parts.append(obs['robot0_gripper_qpos'])
        
        # Joint positions
        if 'robot0_joint_pos' in obs:
            proprio_parts.append(obs['robot0_joint_pos'])
        
        # Joint velocities
        if 'robot0_joint_vel' in obs:
            proprio_parts.append(obs['robot0_joint_vel'])
        
        if not proprio_parts:
            raise ValueError(f"No proprio found in obs. Keys: {obs.keys()}")
        
        return np.concatenate(proprio_parts)
    
    @torch.no_grad()
    def run_episode(
        self,
        env,
        goal_image: np.ndarray,
        verbose: bool = False,
    ) -> Dict:
        """
        Run a single evaluation episode.
        
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
        
        # Convert goal image to tensor
        goal_tensor = torch.from_numpy(goal_image).permute(2, 0, 1).float() / 255.0
        goal_tensor = goal_tensor.unsqueeze(0).to(self.device)
        
        # Pre-compute goal embedding
        goal_emb = self.model.vjepa2.encode_image(goal_tensor)
        
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
                
                proprio = np.stack(proprio_buffer)  # (history, proprio_dim)
                proprio_tensor = torch.from_numpy(proprio).float()
                proprio_tensor = proprio_tensor.unsqueeze(0).to(self.device)  # (1, history, dim)
                
                # Get action chunk from policy
                action_chunk = self.model(video_tensor, goal_tensor, proprio_tensor)
                action_chunk = action_chunk[0].cpu().numpy()  # (chunk_size, action_dim)
                
                # Fill action buffer (take execute_steps actions)
                action_buffer = list(action_chunk[:self.execute_steps])
            
            # Execute action
            action = action_buffer.pop(0)
            all_actions.append(action)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if verbose and step % 50 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            
            # Check success
            if done or info.get('success', False):
                success = info.get('success', False)
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
        n_episodes: int = 20,
        verbose: bool = False,
    ) -> Dict:
        """
        Evaluate on a single task.
        
        Returns:
            dict with success_rate, avg_reward, avg_length, per_episode results
        """
        env = self.create_env(task)
        goal_image = self.get_goal_image(env, task)
        
        results = []
        
        for ep in range(n_episodes):
            if verbose:
                print(f"  Episode {ep + 1}/{n_episodes}")
            
            result = self.run_episode(env, goal_image, verbose=verbose)
            results.append(result)
            
            if verbose:
                status = "✅" if result['success'] else "❌"
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
            suite_name: One of libero_object, libero_spatial, libero_goal, libero_long
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
            task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"
            
            if verbose:
                print(f"\nTask {task_idx + 1}/{n_tasks}: {task_name}")
            
            result = self.evaluate_task(
                task,
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
        n_episodes_per_task: int = 20,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate on all LIBERO suites.
        
        Args:
            n_episodes_per_task: Episodes per task
            save_path: Path to save results JSON
        
        Returns:
            dict with results for all suites
        """
        suites = [
            'libero_object',
            'libero_spatial',
            'libero_goal',
            'libero_long',
        ]
        
        all_results = {}
        
        for suite in suites:
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
    
    def evaluate_all_suites(self, **kwargs) -> Dict:
        """Return dummy results for all suites"""
        return {
            suite: self.evaluate_suite(suite)
            for suite in ['libero_object', 'libero_spatial', 'libero_goal', 'libero_long']
        }


def create_evaluator(model, **kwargs):
    """Factory function to create appropriate evaluator"""
    if LIBERO_AVAILABLE:
        return LIBEROEvaluator(model, **kwargs)
    else:
        return DummyEvaluator(model, **kwargs)