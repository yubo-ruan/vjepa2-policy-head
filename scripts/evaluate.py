#!/usr/bin/env python3
"""
Unified Evaluation Script for V-JEPA 2 Policy.

This script evaluates trained policies on the LIBERO benchmark suites by
running episodes in the simulation environment and measuring success rates.

Evaluation Pipeline:
    1. Load trained model checkpoint
    2. Initialize LIBERO environment for specified suite
    3. For each task, run N episodes:
       a. Reset environment to random initial state
       b. Execute policy with receding horizon control
       c. Record success/failure
    4. Report per-task and overall success rates

Receding Horizon Control:
    - At each decision point, encode current observation (16 frames) with V-JEPA 2
    - Policy predicts action chunk of 50 actions
    - Execute first `execute_steps` actions (default: 10)
    - Repeat until episode ends or max steps reached

Usage:
    # Basic evaluation on single suite
    python scripts/evaluate.py --checkpoint best_model.pt --suite libero_spatial

    # Evaluate specific tasks with fewer episodes
    python scripts/evaluate.py --checkpoint best_model.pt --suite libero_spatial --tasks 0 1 2 --n_episodes 10

    # Evaluate all suites
    python scripts/evaluate.py --checkpoint best_model.pt --suite all

    # Save evaluation videos
    python scripts/evaluate.py --checkpoint best_model.pt --suite libero_spatial --save_videos

CLI Arguments:
    --checkpoint: Path to trained model checkpoint (required)
    --suite: LIBERO suite to evaluate (libero_spatial, libero_object, etc., or "all")
    --tasks: Specific task indices to evaluate (default: all tasks in suite)
    --n_episodes: Number of episodes per task (default: 20)
    --execute_steps: Actions to execute before replanning (default: 10)
    --max_steps: Maximum steps per episode (default: 300)
    --save_videos: Save episode videos as GIFs
    --video_dir: Directory for saved videos
    --save_results: Path to save results JSON
    --device: Device to run on (cuda/cpu)
"""

import argparse
import torch
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vjepa_policy.models.policy import VJEPA2Policy
from vjepa_policy.utils.evaluation import LIBEROEvaluator, VALID_SUITES


def main():
    # ========== Parse Arguments ==========
    parser = argparse.ArgumentParser(description='Evaluate V-JEPA 2 Policy')

    # Required argument
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

    # Output options
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

    # ========== Load Checkpoint ==========
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract configuration from checkpoint
    # (Training script saves config alongside model weights)
    config = checkpoint.get('config', {})
    model_cfg = config.get('model', {})
    encoder_cfg = config.get('encoder', {})
    data_cfg = config.get('data', {})

    # ========== Create Model ==========
    # Note: V-JEPA 2 encoder will be lazy-loaded on first use
    # This is required for evaluation (unlike training which uses precomputed embeddings)
    print("Creating model...")
    model = VJEPA2Policy(
        vjepa2_model_path=encoder_cfg.get('model_path', '/workspace/models/vjepa2-ac-vitg.pt'),
        vjepa2_model_name=encoder_cfg.get('model_name', 'vjepa2_vitg'),
        vjepa2_freeze=True,  # Always frozen during evaluation
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

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    # ========== Determine Suites to Evaluate ==========
    if args.suite == 'all':
        # Evaluate on multiple suites if specified in config, otherwise use defaults
        suites = config.get('evaluation', {}).get('suites', ['libero_object', 'libero_spatial', 'libero_goal'])
    else:
        suites = [args.suite]

    all_results = {}

    # ========== Run Evaluation ==========
    for suite in suites:
        # Create evaluator for this suite
        # LIBEROEvaluator handles environment creation, episode running, and success detection
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

        # Print evaluation configuration
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

        # Run evaluation
        results = evaluator.evaluate_suite(
            suite_name=suite,
            n_episodes=args.n_episodes,
            task_indices=args.tasks,
            save_videos=args.save_videos,
            video_dir=args.video_dir,
        )

        all_results[suite] = results

        # ========== Print Results for This Suite ==========
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

    # ========== Print Final Summary ==========
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

    # ========== Save Results ==========
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
