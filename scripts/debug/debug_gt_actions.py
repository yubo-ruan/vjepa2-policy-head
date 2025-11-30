"""
DEBUG: Execute ground truth actions directly in LIBERO environment.

This is a critical sanity check:
- If GT actions succeed -> model prediction is wrong
- If GT actions fail -> execution pipeline is wrong (or action format mismatch)

This will tell us if the problem is in our model predictions or somewhere else.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# LIBERO imports
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def load_demo_actions(task, demo_dir: str = "/workspace/data/libero", demo_idx: int = 0):
    """Load actions from a successful demo"""
    suite_dir = Path(demo_dir) / task.problem_folder
    demo_file = suite_dir / f"{task.name}_demo.hdf5"

    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        return None

    with h5py.File(demo_file, 'r') as f:
        actions = f[f'data/demo_{demo_idx}/actions'][:]

        # Also get info about the demo
        n_steps = actions.shape[0]
        print(f"Demo {demo_idx}: {n_steps} steps")
        print(f"Action shape: {actions.shape}")
        print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"Action mean: {actions.mean(axis=0)}")
        print(f"Action std: {actions.std(axis=0)}")

        return actions


def replay_demo_actions(env, actions, verbose=True):
    """
    Replay ground truth actions in the environment.
    Returns success and episode info.
    """
    obs = env.reset()

    total_reward = 0
    success = False

    for step, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose and step % 50 == 0:
            print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            print(f"    Action: {action[:3]} (pos) {action[3:6]} (rot) {action[6]:.2f} (grip)")

        if done or info.get('success', False):
            success = info.get('success', False)
            if verbose:
                print(f"  Episode done at step {step}: success={success}")
            break

    return {
        'success': success,
        'total_reward': total_reward,
        'episode_length': step + 1,
    }


def main():
    print("="*60)
    print("GROUND TRUTH ACTION REPLAY TEST")
    print("="*60)
    print()

    # Setup
    suite_name = "libero_spatial"
    image_size = 256

    # Get benchmark
    BenchClass = benchmark.get_benchmark(suite_name)
    bench = BenchClass()
    n_tasks = bench.get_num_tasks()

    print(f"Suite: {suite_name}")
    print(f"Number of tasks: {n_tasks}")
    print()

    # Test first few tasks
    n_test_tasks = 3
    n_demos_per_task = 2

    results = []

    for task_idx in range(min(n_test_tasks, n_tasks)):
        task = bench.get_task(task_idx)
        bddl_file_path = bench.get_task_bddl_file_path(task_idx)
        task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"

        print(f"\n{'='*60}")
        print(f"Task {task_idx}: {task_name}")
        print(f"{'='*60}")

        # Create environment
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_path,
            render_camera="agentview",
            camera_heights=image_size,
            camera_widths=image_size,
        )

        # Test with multiple demos
        task_successes = []

        for demo_idx in range(n_demos_per_task):
            print(f"\n--- Demo {demo_idx} ---")

            # Load demo actions
            actions = load_demo_actions(task, demo_idx=demo_idx)
            if actions is None:
                continue

            # Replay actions
            result = replay_demo_actions(env, actions, verbose=True)
            task_successes.append(result['success'])

            status = "SUCCESS" if result['success'] else "FAIL"
            print(f"\nResult: {status}")
            print(f"Total reward: {result['total_reward']:.2f}")
            print(f"Episode length: {result['episode_length']}")

        env.close()

        if task_successes:
            task_success_rate = np.mean(task_successes)
            results.append(task_success_rate)
            print(f"\nTask success rate: {task_success_rate*100:.0f}%")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if results:
        overall = np.mean(results)
        print(f"Overall GT action success rate: {overall*100:.1f}%")

        if overall > 0.5:
            print("\n>>> GT actions WORK! Problem is in model predictions.")
        else:
            print("\n>>> GT actions FAIL! Problem is in execution pipeline or action format.")
            print("    This could mean:")
            print("    1. Environment randomization differs from demo recording")
            print("    2. Action space interpretation mismatch")
            print("    3. Physics/robot config differences")
    else:
        print("No results collected")


if __name__ == "__main__":
    main()
