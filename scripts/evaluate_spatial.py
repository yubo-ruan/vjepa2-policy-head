"""
Evaluation script for V-JEPA 2 Policy with SPATIAL tokens on LIBERO benchmark

Uses VJEPA2PolicySpatial model with 64 spatial tokens per modality.
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2PolicySpatial
from vjepa_policy.utils.evaluation import LIBEROEvaluatorSpatial, VALID_SUITES


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    # Load config
    config = load_config(args.config)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get spatial config
    n_spatial_tokens = config.get('spatial', {}).get('n_tokens', 64)

    # Create spatial model
    print("Creating VJEPA2PolicySpatial model...")
    model = VJEPA2PolicySpatial(
        vjepa2_model_path=args.model_path or config['vjepa2']['model_path'],
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
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create evaluator
    normalize_embeddings = config.get('robust_embeddings', {}).get('normalize', True)

    evaluator = LIBEROEvaluatorSpatial(
        model=model,
        device=device,
        video_len=config['vjepa2']['num_frames'],
        proprio_history=config['proprio']['history_len'],
        chunk_size=config['policy']['chunk_size'],
        execute_steps=config['evaluation']['execute_steps'],
        max_episode_steps=config['evaluation']['max_episode_steps'],
        image_size=config['vjepa2']['image_size'],
        normalize_embeddings=normalize_embeddings,
        n_spatial_tokens=n_spatial_tokens,
    )

    # Run evaluation
    if args.suite == 'all':
        suites = config['evaluation'].get('suites', ['libero_object', 'libero_spatial', 'libero_goal'])
        results = evaluator.evaluate_all_suites(
            suites=suites,
            n_episodes_per_task=args.n_episodes,
            save_path=args.save_results,
        )
    else:
        results = evaluator.evaluate_suite(
            suite_name=args.suite,
            n_episodes_per_task=args.n_episodes,
            verbose=True,
        )

        # Save single suite results
        if args.save_results:
            import json
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    print("\nEvaluation complete!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA 2 Policy (Spatial) on LIBERO")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/spatial.yaml",
                        help="Path to config file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to V-JEPA 2 weights (overrides config)")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_90', 'libero_10', 'all'],
                        help="Which LIBERO suite to evaluate")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of episodes per task")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()
    main(args)
