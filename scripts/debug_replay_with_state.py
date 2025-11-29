"""
DEBUG: Replay demo actions with RESTORED initial state.

This tests if demo actions work when we restore the exact initial state.
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


def replay_with_restored_state(
    task,
    bddl_file_path: str,
    demo_dir: str = "/workspace/data/libero",
    demo_idx: int = 0,
    verbose: bool = True
):
    """Replay demo with restored initial state"""
    suite_dir = Path(demo_dir) / task.problem_folder
    demo_file = suite_dir / f"{task.name}_demo.hdf5"

    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        return None

    # Load demo data
    with h5py.File(demo_file, 'r') as f:
        actions = f[f'data/demo_{demo_idx}/actions'][:]
        states = f[f'data/demo_{demo_idx}/states'][:]
        initial_state = states[0]

    print(f"Actions: {actions.shape}")
    print(f"States: {states.shape}")
    print(f"Initial state: {initial_state.shape}")

    # Create environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        render_camera="agentview",
        camera_heights=256,
        camera_widths=256,
    )

    # Reset and restore initial state
    obs = env.reset()

    # Try to restore initial state
    print("\nRestoring initial state...")
    try:
        # Access the underlying robosuite environment
        if hasattr(env.env, 'sim'):
            env.env.sim.set_state_from_flattened(initial_state)
            env.env.sim.forward()  # Step physics to update
            print("Successfully restored initial state via sim.set_state_from_flattened()")
        else:
            print("Cannot access env.env.sim")
            env.close()
            return None
    except Exception as e:
        print(f"Error restoring state: {e}")
        env.close()
        return None

    # Get observation after state restoration
    obs = env.env._get_observations()

    # Replay actions
    print(f"\nReplaying {len(actions)} actions...")
    total_reward = 0
    success = False

    for step, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose and step % 20 == 0:
            print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            print(f"    Action: {action[:3]} (pos) {action[3:6]} (rot) {action[6]:.2f} (grip)")

        if done or info.get('success', False):
            success = info.get('success', False)
            print(f"  Done at step {step}, success={success}")
            break

    env.close()

    return {
        'success': success,
        'total_reward': total_reward,
        'episode_length': step + 1,
    }


def main():
    print("="*60)
    print("REPLAY WITH RESTORED INITIAL STATE")
    print("="*60)
    print()

    suite_name = "libero_spatial"

    # Get benchmark
    BenchClass = benchmark.get_benchmark(suite_name)
    bench = BenchClass()
    n_tasks = bench.get_num_tasks()

    print(f"Suite: {suite_name}")
    print(f"Number of tasks: {n_tasks}")

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

        task_successes = []

        for demo_idx in range(n_demos_per_task):
            print(f"\n--- Demo {demo_idx} ---")

            result = replay_with_restored_state(
                task,
                bddl_file_path,
                demo_idx=demo_idx,
                verbose=True
            )

            if result:
                task_successes.append(result['success'])
                status = "SUCCESS" if result['success'] else "FAIL"
                print(f"\nResult: {status}")
                print(f"Total reward: {result['total_reward']:.2f}")

        if task_successes:
            task_rate = np.mean(task_successes)
            results.append(task_rate)
            print(f"\nTask success rate: {task_rate*100:.0f}%")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if results:
        overall = np.mean(results)
        print(f"Overall success rate with restored state: {overall*100:.1f}%")

        if overall > 0.8:
            print("\n>>> Demo replay with restored state WORKS!")
            print("    The actions are correct. Problem is environment randomization.")
            print("    Policy must learn to generalize to different initial states.")
        elif overall > 0.5:
            print("\n>>> Demo replay partially works.")
            print("    There may be some physics instability or state restoration issues.")
        else:
            print("\n>>> Demo replay with restored state still fails.")
            print("    This suggests deeper issues with action format or environment.")


if __name__ == "__main__":
    main()
