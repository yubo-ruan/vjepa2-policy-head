"""
DEBUG: Check if LIBERO demos have saved initial states.

The issue: We're replaying actions from demos, but the environment
randomizes the initial state. We need to restore the exact initial
state from the demo for actions to work.
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


def inspect_demo_file(task, demo_dir: str = "/workspace/data/libero"):
    """Inspect what's stored in demo HDF5 file"""
    suite_dir = Path(demo_dir) / task.problem_folder
    demo_file = suite_dir / f"{task.name}_demo.hdf5"

    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        return

    print(f"\nInspecting: {demo_file}")

    with h5py.File(demo_file, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  DATASET: {name}, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  GROUP: {name}")

        print("\nFile structure:")
        f.visititems(print_structure)

        # Check for initial state
        print("\n" + "="*50)
        print("Looking for initial state info...")

        # Check common locations for initial state
        possible_state_keys = [
            'data/demo_0/states',
            'data/demo_0/initial_state',
            'data/demo_0/model_file',
            'data/attrs/env_args',
            'env_args',
        ]

        for key in possible_state_keys:
            if key in f:
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    print(f"Found {key}: shape={data.shape}, dtype={data.dtype}")
                    if data.shape[0] < 5:
                        print(f"  Value: {data[:]}")
                else:
                    print(f"Found {key}: (group)")

        # Check attributes
        print("\nRoot attributes:")
        for key in f.attrs.keys():
            val = f.attrs[key]
            if isinstance(val, bytes):
                val = val.decode('utf-8') if len(val) < 200 else f"<{len(val)} bytes>"
            print(f"  {key}: {val}")

        # Check demo_0 attributes
        if 'data/demo_0' in f:
            print("\nDemo 0 attributes:")
            for key in f['data/demo_0'].attrs.keys():
                val = f['data/demo_0'].attrs[key]
                if isinstance(val, bytes):
                    val = val.decode('utf-8') if len(val) < 200 else f"<{len(val)} bytes>"
                print(f"  {key}: {val}")

        # Check if states exist
        if 'data/demo_0/states' in f:
            states = f['data/demo_0/states'][:]
            print(f"\nStates shape: {states.shape}")
            print(f"Initial state (states[0]): shape={states[0].shape}")


def replay_with_initial_state(task, demo_dir: str = "/workspace/data/libero", demo_idx: int = 0):
    """Try to replay with restored initial state"""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite_dir = Path(demo_dir) / task.problem_folder
    demo_file = suite_dir / f"{task.name}_demo.hdf5"

    if not demo_file.exists():
        print(f"Demo file not found: {demo_file}")
        return

    with h5py.File(demo_file, 'r') as f:
        # Get actions
        actions = f[f'data/demo_{demo_idx}/actions'][:]

        # Try to get initial state
        initial_state = None
        if f'data/demo_{demo_idx}/states' in f:
            initial_state = f[f'data/demo_{demo_idx}/states'][0]
            print(f"Got initial state: shape={initial_state.shape}")

    # Create environment
    # Get bddl file from benchmark
    BenchClass = benchmark.get_benchmark('libero_spatial')
    bench = BenchClass()

    for task_idx in range(bench.get_num_tasks()):
        t = bench.get_task(task_idx)
        if t.name == task.name:
            bddl_file_path = bench.get_task_bddl_file_path(task_idx)
            break

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        render_camera="agentview",
        camera_heights=256,
        camera_widths=256,
    )

    # Try to set initial state
    obs = env.reset()

    if initial_state is not None:
        print("\nTrying to set initial state...")
        try:
            # LIBERO environments should have set_init_state
            if hasattr(env.env, 'sim') and hasattr(env.env.sim, 'set_state_from_flattened'):
                env.env.sim.set_state_from_flattened(initial_state)
                env.env.sim.forward()
                print("Set initial state successfully!")
            else:
                print("Environment doesn't support state restoration via sim")

                # Alternative: Try env.reset with init_state
                print("Checking if env supports init_state in reset...")
                print(f"env type: {type(env)}")
                print(f"env.env type: {type(env.env)}")
        except Exception as e:
            print(f"Error setting state: {e}")

    # Replay actions
    print(f"\nReplaying {len(actions)} actions...")
    total_reward = 0
    for step, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step % 30 == 0:
            print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
        if done or info.get('success', False):
            print(f"  Done at step {step}, success={info.get('success', False)}")
            break

    env.close()
    print(f"\nFinal reward: {total_reward}")


def main():
    print("="*60)
    print("INVESTIGATING LIBERO DEMO INITIAL STATE")
    print("="*60)

    # Get a task
    BenchClass = benchmark.get_benchmark('libero_spatial')
    bench = BenchClass()
    task = bench.get_task(0)

    print(f"Task: {task.name}")

    # Inspect demo file
    inspect_demo_file(task)

    # Try replay with initial state
    print("\n" + "="*60)
    print("ATTEMPTING REPLAY WITH INITIAL STATE")
    print("="*60)
    replay_with_initial_state(task)


if __name__ == "__main__":
    main()
