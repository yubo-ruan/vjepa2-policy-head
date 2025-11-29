"""
Test receding horizon evaluation with different execute_steps values.

Compares:
- execute_steps=10 (current default): Execute 10 actions before replanning
- execute_steps=1 (receding horizon): Re-predict every step for maximum error correction
- execute_steps=5 (middle ground): Balance between efficiency and correction
"""

import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2PolicySpatial
from vjepa_policy.utils.evaluation import LIBEROEvaluatorSpatial

# LIBERO imports
from libero.libero import benchmark


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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model, config


def test_execute_steps(
    model,
    config: dict,
    suite_name: str,
    execute_steps_list: list = [10, 5, 1],
    n_tasks: int = 3,
    n_episodes: int = 5,
    device: str = "cuda",
):
    """Test different execute_steps values"""

    results = {}

    for execute_steps in execute_steps_list:
        print(f"\n{'='*60}")
        print(f"Testing execute_steps = {execute_steps}")
        print(f"{'='*60}")

        # Create evaluator with specific execute_steps
        evaluator = LIBEROEvaluatorSpatial(
            model=model,
            device=device,
            video_len=config['vjepa2']['num_frames'],
            proprio_history=config['proprio']['history_len'],
            chunk_size=config['policy']['chunk_size'],
            execute_steps=execute_steps,  # KEY PARAMETER
            max_episode_steps=config['evaluation']['max_episode_steps'],
            image_size=config['vjepa2']['image_size'],
            normalize_embeddings=config['robust_embeddings']['normalize'],
            n_spatial_tokens=config['spatial']['n_tokens'],
        )

        # Get benchmark
        BenchClass = benchmark.get_benchmark(suite_name)
        bench = BenchClass()

        all_successes = []
        all_rewards = []

        for task_idx in range(min(n_tasks, bench.get_num_tasks())):
            task = bench.get_task(task_idx)
            bddl_file_path = bench.get_task_bddl_file_path(task_idx)
            task_name = task.language if hasattr(task, 'language') else f"task_{task_idx}"

            print(f"\nTask {task_idx}: {task_name[:50]}...")

            result = evaluator.evaluate_task(
                task=task,
                bddl_file_path=bddl_file_path,
                n_episodes=n_episodes,
                verbose=False,
            )

            all_successes.extend([r['success'] for r in result['per_episode']])
            all_rewards.extend([r['total_reward'] for r in result['per_episode']])

            print(f"  Success rate: {result['success_rate']*100:.0f}%")
            print(f"  Avg reward: {result['avg_reward']:.3f}")

        # Store results
        results[execute_steps] = {
            'success_rate': np.mean(all_successes),
            'avg_reward': np.mean(all_rewards),
            'n_successes': sum(all_successes),
            'n_episodes': len(all_successes),
        }

        print(f"\nexecute_steps={execute_steps} Overall: {np.mean(all_successes)*100:.1f}% success")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='/workspace/checkpoints_spatial/best_model.pt')
    parser.add_argument('--config', type=str,
                       default='configs/spatial.yaml')
    parser.add_argument('--suite', type=str, default='libero_spatial')
    parser.add_argument('--n_tasks', type=int, default=3,
                       help='Number of tasks to test (for speed)')
    parser.add_argument('--n_episodes', type=int, default=5,
                       help='Episodes per task')
    parser.add_argument('--execute_steps', type=str, default='10,5,1',
                       help='Comma-separated execute_steps values to test')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Parse execute_steps
    execute_steps_list = [int(x) for x in args.execute_steps.split(',')]
    print(f"Testing execute_steps: {execute_steps_list}")

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)

    # Run comparison
    results = test_execute_steps(
        model=model,
        config=config,
        suite_name=args.suite,
        execute_steps_list=execute_steps_list,
        n_tasks=args.n_tasks,
        n_episodes=args.n_episodes,
        device=device,
    )

    # Summary
    print(f"\n{'='*60}")
    print("RECEDING HORIZON COMPARISON")
    print(f"{'='*60}")
    print(f"Suite: {args.suite}")
    print(f"Tasks: {args.n_tasks}, Episodes/task: {args.n_episodes}")
    print()

    print(f"{'execute_steps':<15} {'Success Rate':<15} {'Avg Reward':<15}")
    print("-" * 45)
    for steps, r in sorted(results.items(), key=lambda x: -x[0]):
        print(f"{steps:<15} {r['success_rate']*100:>10.1f}%     {r['avg_reward']:>10.3f}")

    # Recommendation
    best_steps = max(results.keys(), key=lambda x: results[x]['success_rate'])
    print(f"\nBest execute_steps: {best_steps}")

    if best_steps == 1:
        print("Receding horizon (execute_steps=1) performs best!")
        print("Consider updating configs/spatial.yaml with: execute_steps: 1")
    elif best_steps < 10:
        print(f"Smaller execute_steps ({best_steps}) helps with generalization.")


if __name__ == "__main__":
    main()
