"""
Overfit Test with Real LIBERO Data

This test verifies:
1. Dataset loader works correctly
2. Data shapes are compatible with model
3. Model can memorize real data samples
4. Full pipeline (V-JEPA 2 encoder -> policy head) works

If this fails, there's a bug in data preprocessing or model integration.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2Policy
from vjepa_policy.data.libero_dataset import LIBERODataset


def test_libero_data_shapes():
    """Test that LIBERO data shapes are compatible with model"""
    print("=" * 60)
    print("TEST 1: LIBERO Data Shape Compatibility")
    print("=" * 60)

    data_dir = "/workspace/data/libero"

    if not Path(data_dir).exists():
        print(f"LIBERO data not found at {data_dir}")
        return False

    # Load dataset
    dataset = LIBERODataset(
        data_dir=data_dir,
        suite="libero_object",
        split="train",
        video_len=16,
        proprio_history=5,
        chunk_size=50,
        image_size=256,  # Resize for V-JEPA 2
    )

    if len(dataset) == 0:
        print("No samples in dataset!")
        return False

    sample = dataset[0]

    print(f"\nSample shapes:")
    print(f"  video: {sample['video'].shape}")  # Should be (16, 3, 256, 256)
    print(f"  goal: {sample['goal'].shape}")    # Should be (3, 256, 256)
    print(f"  proprio: {sample['proprio'].shape}")  # Should be (5, 15)
    print(f"  actions: {sample['actions'].shape}")  # Should be (50, 7)

    # Verify shapes
    assert sample['video'].shape == (16, 3, 256, 256), f"Wrong video shape: {sample['video'].shape}"
    assert sample['goal'].shape == (3, 256, 256), f"Wrong goal shape: {sample['goal'].shape}"
    assert sample['proprio'].shape[0] == 5, f"Wrong proprio history: {sample['proprio'].shape}"
    assert sample['actions'].shape == (50, 7), f"Wrong actions shape: {sample['actions'].shape}"

    # Verify value ranges
    print(f"\nValue ranges:")
    print(f"  video: [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
    print(f"  goal: [{sample['goal'].min():.3f}, {sample['goal'].max():.3f}]")
    print(f"  proprio: [{sample['proprio'].min():.3f}, {sample['proprio'].max():.3f}]")
    print(f"  actions: [{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]")

    # Images should be in [0, 1]
    assert sample['video'].min() >= 0 and sample['video'].max() <= 1, "Video not normalized!"
    assert sample['goal'].min() >= 0 and sample['goal'].max() <= 1, "Goal not normalized!"

    print("\n✅ Data shape test passed!")
    return True


def test_libero_with_encoder():
    """Test LIBERO data with V-JEPA 2 encoder"""
    print("\n" + "=" * 60)
    print("TEST 2: LIBERO Data with V-JEPA 2 Encoder")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = "/workspace/data/libero"

    # Load dataset
    dataset = LIBERODataset(
        data_dir=data_dir,
        suite="libero_object",
        split="train",
        video_len=16,
        proprio_history=5,
        chunk_size=50,
        image_size=256,
    )

    # Get a few samples
    n_samples = 4
    samples = [dataset[i] for i in range(n_samples)]

    # Stack into batch
    video = torch.stack([s['video'] for s in samples]).to(device)  # (B, T, C, H, W)
    goal = torch.stack([s['goal'] for s in samples]).to(device)    # (B, C, H, W)
    proprio = torch.stack([s['proprio'] for s in samples]).to(device)
    actions = torch.stack([s['actions'] for s in samples]).to(device)

    print(f"\nBatch shapes:")
    print(f"  video: {video.shape}")
    print(f"  goal: {goal.shape}")
    print(f"  proprio: {proprio.shape}")
    print(f"  actions: {actions.shape}")

    # Create model
    print("\nLoading model...")
    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        vjepa2_freeze=True,
        proprio_dim=15,  # LIBERO proprio dim
        device=device,
    )

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        pred_actions = model(video, goal, proprio)

    print(f"\nOutput:")
    print(f"  pred_actions: {pred_actions.shape}")
    print(f"  action range: [{pred_actions.min():.3f}, {pred_actions.max():.3f}]")

    assert pred_actions.shape == actions.shape, f"Shape mismatch: {pred_actions.shape} vs {actions.shape}"

    print("\n✅ Encoder integration test passed!")
    return True


def test_libero_overfit(
    n_samples: int = 10,
    n_epochs: int = 500,
    lr: float = 1e-3,
    target_loss: float = 0.15,
):
    """
    Overfit on real LIBERO data.

    This catches data preprocessing bugs that synthetic data wouldn't catch.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Overfit on Real LIBERO Data")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = "/workspace/data/libero"

    # Load dataset
    print("Loading dataset...")
    dataset = LIBERODataset(
        data_dir=data_dir,
        suite="libero_object",
        split="train",
        video_len=16,
        proprio_history=5,
        chunk_size=50,
        image_size=256,
    )

    # Get samples
    print(f"Loading {n_samples} samples...")
    samples = [dataset[i] for i in range(min(n_samples, len(dataset)))]

    # Stack into batch
    video = torch.stack([s['video'] for s in samples]).to(device)
    goal = torch.stack([s['goal'] for s in samples]).to(device)
    proprio = torch.stack([s['proprio'] for s in samples]).to(device)
    target_actions = torch.stack([s['actions'] for s in samples]).to(device)

    print(f"\nBatch shapes:")
    print(f"  video: {video.shape}")
    print(f"  goal: {goal.shape}")
    print(f"  proprio: {proprio.shape}")
    print(f"  target_actions: {target_actions.shape}")

    # Create model
    print("\nLoading model...")
    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        vjepa2_freeze=True,
        proprio_dim=15,
        policy_hidden_dim=512,
        policy_n_layers=4,
        policy_n_context_tokens=4,
        device=device,
    )

    # Pre-compute embeddings (faster training)
    print("Pre-computing V-JEPA 2 embeddings...")
    with torch.no_grad():
        current_emb = model.vjepa2.encode_video(video)
        goal_emb = model.vjepa2.encode_image(goal)

    print(f"  current_emb: {current_emb.shape}")
    print(f"  goal_emb: {goal_emb.shape}")

    # Optimizer
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=lr)

    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"Target loss: {target_loss}")

    losses = []

    for epoch in tqdm(range(n_epochs)):
        # Forward
        pred_actions = model.forward_with_precomputed(current_emb, goal_emb, proprio)

        # Loss
        loss = F.l1_loss(pred_actions, target_actions)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    final_loss = losses[-1]

    # Check result
    print(f"\n{'='*50}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Target: {target_loss}")

    # Also check per-sample loss
    with torch.no_grad():
        pred_actions = model.forward_with_precomputed(current_emb, goal_emb, proprio)
        per_sample_loss = F.l1_loss(pred_actions, target_actions, reduction='none').mean(dim=[1, 2])
        print(f"\nPer-sample losses:")
        for i, l in enumerate(per_sample_loss):
            print(f"  Sample {i}: {l.item():.4f}")

    if final_loss < target_loss:
        print("\n✅ OVERFIT TEST PASSED!")
        print("Model can memorize real LIBERO data - pipeline is working!")
        return True
    else:
        print("\n❌ OVERFIT TEST FAILED!")
        print("Model cannot memorize real LIBERO data - check preprocessing!")
        return False


def run_all_tests():
    """Run all LIBERO tests"""
    print("\n" + "=" * 60)
    print("LIBERO DATA INTEGRATION TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Data shapes
    results['data_shapes'] = test_libero_data_shapes()

    # Test 2: Encoder integration
    results['encoder'] = test_libero_with_encoder()

    # Test 3: Overfit
    results['overfit'] = test_libero_overfit()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL LIBERO TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["shapes", "encoder", "overfit", "all"], default="all")
    args = parser.parse_args()

    if args.test == "shapes":
        test_libero_data_shapes()
    elif args.test == "encoder":
        test_libero_with_encoder()
    elif args.test == "overfit":
        test_libero_overfit()
    else:
        run_all_tests()
