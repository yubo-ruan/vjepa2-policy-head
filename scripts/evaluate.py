#!/usr/bin/env python3
"""
Unified Evaluation Script for V-JEPA 2 Policy.

Evaluates trained policy on LIBERO benchmark suites.

Usage:
    python scripts/evaluate.py --checkpoint best_model.pt --suite libero_spatial
    python scripts/evaluate.py --checkpoint best_model.pt --suite libero_spatial --n_episodes 10
    python scripts/evaluate.py --checkpoint best_model.pt --suite libero_spatial --save_videos
"""

import argparse
import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.policy import VJEPA2Policy
from vjepa_policy.utils.evaluation import LIBEROEvaluator, VALID_SUITES


def main():
    parser = argparse.ArgumentParser(description='Evaluate V-JEPA 2 Policy')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')

    # Evaluation settings
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=VALID_SUITES + ['all'],
                        help='LIBERO suite to evaluate on (or "all")')
    parser.add_argument('--tasks', type=int, nargs='+', default=None,
                        help='Specific task indices (default: all)')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes per task')
    parser.add_argument('--execute_steps', type=int, default=10,
                        help='Actions to execute before replanning')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Max steps per episode')

    # Output
    parser.add_argument('--save_videos', action='store_true',
                        help='Save evaluation videos')
    parser.add_argument('--video_dir', type=str, default='videos',
                        help='Directory to save videos')
    parser.add_argument('--save_results', type=str, default=None,
                        help='File to save results JSON')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get config from checkpoint
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
        device=args.device,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    # Determine suites to evaluate
    if args.suite == 'all':
        suites = config.get('evaluation', {}).get('suites', ['libero_object', 'libero_spatial', 'libero_goal'])
    else:
        suites = [args.suite]

    all_results = {}

    for suite in suites:
        # Create evaluator
        print(f"\nCreating evaluator for {suite}...")
        evaluator = LIBEROEvaluator(
            model=model,
            device=args.device,
            video_len=encoder_cfg.get('num_frames', 16),
            proprio_history=model_cfg.get('proprio_history', 5),
            chunk_size=model_cfg.get('chunk_size', 50),
            execute_steps=args.execute_steps,
            max_episode_steps=args.max_steps,
            normalize_embeddings=data_cfg.get('normalize', True),
        )

        # Run evaluation
        print()
        print("=" * 60)
        print(f"Evaluating on {suite}")
        print(f"Episodes per task: {args.n_episodes}")
        print(f"Execute steps: {args.execute_steps}")
        print(f"Max steps: {args.max_steps}")
        if args.tasks:
            print(f"Tasks: {args.tasks}")
        print("=" * 60)
        print()

        results = evaluator.evaluate_suite(
            suite_name=suite,
            n_episodes=args.n_episodes,
            task_indices=args.tasks,
            save_videos=args.save_videos,
            video_dir=args.video_dir,
        )

        all_results[suite] = results

        # Print results for this suite
        print()
        print("-" * 40)
        print(f"Results for {suite}")
        print("-" * 40)

        total_success = 0
        total_episodes = 0

        for task_idx in sorted(results.keys()):
            task_results = results[task_idx]
            successes = task_results.get('successes', task_results.get('success', []))
            success = sum(successes)
            n = len(successes)
            rate = 100 * success / n if n > 0 else 0
            print(f"  Task {task_idx}: {success}/{n} ({rate:.1f}%)")
            total_success += success
            total_episodes += n

        overall = 100 * total_success / total_episodes if total_episodes > 0 else 0
        print(f"  Overall: {total_success}/{total_episodes} ({overall:.1f}%)")

    # Print final summary
    print()
    print("=" * 60)
    print("Final Summary")
    print("=" * 60)

    grand_total_success = 0
    grand_total_episodes = 0

    for suite, results in all_results.items():
        total_success = 0
        total_episodes = 0
        for task_results in results.values():
            successes = task_results.get('successes', task_results.get('success', []))
            total_success += sum(successes)
            total_episodes += len(successes)
        rate = 100 * total_success / total_episodes if total_episodes > 0 else 0
        print(f"{suite}: {total_success}/{total_episodes} ({rate:.1f}%)")
        grand_total_success += total_success
        grand_total_episodes += total_episodes

    if len(suites) > 1:
        overall = 100 * grand_total_success / grand_total_episodes if grand_total_episodes > 0 else 0
        print(f"\nTotal: {grand_total_success}/{grand_total_episodes} ({overall:.1f}%)")

    # Save results
    if args.save_results:
        results_data = {
            'checkpoint': str(args.checkpoint),
            'suites': suites,
            'n_episodes': args.n_episodes,
            'execute_steps': args.execute_steps,
            'max_steps': args.max_steps,
            'results': {k: {str(tk): tv for tk, tv in v.items()} for k, v in all_results.items()},
        }
        with open(args.save_results, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == '__main__':
    main()
