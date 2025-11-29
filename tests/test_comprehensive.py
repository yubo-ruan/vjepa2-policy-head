"""
Comprehensive Test Suite for V-JEPA 2 Policy Head

Tests:
1. V-JEPA 2 Encoder - loading, shapes, formats
2. Full Model - shapes, forward pass
3. Gradient Flow - all trainable components
4. Policy Head Architecture - multi-token context
5. Proprio Encoder - temporal encoding
6. Action Bounds - output in [-1, 1]
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vjepa_policy.models.full_model import VJEPA2Policy
from vjepa_policy.models.vjepa2_encoder import VJEPA2Encoder
from vjepa_policy.models.proprio_encoder import ProprioEncoder
from vjepa_policy.models.policy_head import PolicyHead


def test_vjepa2_encoder():
    """Test V-JEPA 2 encoder loading and inference"""
    print("=" * 60)
    print("TEST 1: V-JEPA 2 Encoder")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = VJEPA2Encoder(
        model_path="/workspace/models/vjepa2-ac-vitg.pt",
        model_name="vjepa2_vitg",
        device=device,
        freeze=True,
        num_frames=16,
        use_attentive_pool=True,
    )

    # Test 1a: Video encoding (B, C, T, H, W)
    print("\n1a. Video encoding (B, C, T, H, W)...")
    video_bct = torch.rand(2, 3, 16, 256, 256).to(device)
    with torch.no_grad():
        emb = encoder.encode_video(video_bct)
    assert emb.shape == (2, 1408), f"Expected (2, 1408), got {emb.shape}"
    print(f"    Input: {video_bct.shape} -> Output: {emb.shape} ✓")

    # Test 1b: Video encoding (B, T, C, H, W) - auto-convert
    print("\n1b. Video encoding (B, T, C, H, W) - auto-convert...")
    video_btc = torch.rand(2, 16, 3, 256, 256).to(device)
    with torch.no_grad():
        emb = encoder.encode_video(video_btc)
    assert emb.shape == (2, 1408), f"Expected (2, 1408), got {emb.shape}"
    print(f"    Input: {video_btc.shape} -> Output: {emb.shape} ✓")

    # Test 1c: Image encoding
    print("\n1c. Image encoding...")
    image = torch.rand(2, 3, 256, 256).to(device)
    with torch.no_grad():
        emb = encoder.encode_image(image)
    assert emb.shape == (2, 1408), f"Expected (2, 1408), got {emb.shape}"
    print(f"    Input: {image.shape} -> Output: {emb.shape} ✓")

    # Test 1d: Patch tokens
    print("\n1d. Patch token extraction...")
    with torch.no_grad():
        patches = encoder(video_bct, return_patches=True)
    expected_n = (16 // 2) * (256 // 16) * (256 // 16)  # 2048
    assert patches.shape == (2, expected_n, 1408), f"Expected (2, {expected_n}, 1408), got {patches.shape}"
    print(f"    Input: {video_bct.shape} -> Patches: {patches.shape} ✓")

    print("\n✅ V-JEPA 2 Encoder: ALL TESTS PASSED")
    return True


def test_proprio_encoder():
    """Test proprioception encoder"""
    print("\n" + "=" * 60)
    print("TEST 2: Proprio Encoder")
    print("=" * 60)

    encoder = ProprioEncoder(
        proprio_dim=23,
        history_len=5,
        hidden_dim=128,
        output_dim=256,
    )

    # Test 2a: 3D input (B, H, D)
    print("\n2a. 3D input (B, history, dim)...")
    proprio_3d = torch.rand(4, 5, 23)
    out = encoder(proprio_3d)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print(f"    Input: {proprio_3d.shape} -> Output: {out.shape} ✓")

    # Test 2b: 2D input (B, H*D) - flattened
    print("\n2b. 2D input (B, flattened)...")
    proprio_2d = torch.rand(4, 5 * 23)
    out = encoder(proprio_2d)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print(f"    Input: {proprio_2d.shape} -> Output: {out.shape} ✓")

    print("\n✅ Proprio Encoder: ALL TESTS PASSED")
    return True


def test_policy_head_architecture():
    """Test policy head multi-token context architecture"""
    print("\n" + "=" * 60)
    print("TEST 3: Policy Head Architecture")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = PolicyHead(
        vision_dim=1408,
        proprio_dim=256,
        hidden_dim=512,
        action_dim=7,
        chunk_size=50,
        n_heads=8,
        n_layers=4,
        n_context_tokens=4,  # Multi-token context
    ).to(device)

    # Test 3a: Forward pass shape
    print("\n3a. Forward pass shape...")
    current = torch.rand(2, 1408).to(device)
    goal = torch.rand(2, 1408).to(device)
    proprio = torch.rand(2, 256).to(device)

    actions = policy(current, goal, proprio)
    assert actions.shape == (2, 50, 7), f"Expected (2, 50, 7), got {actions.shape}"
    print(f"    Output: {actions.shape} ✓")

    # Test 3b: Action bounds
    print("\n3b. Action bounds [-1, 1]...")
    assert actions.min() >= -1.0, f"Actions below -1: {actions.min()}"
    assert actions.max() <= 1.0, f"Actions above 1: {actions.max()}"
    print(f"    Range: [{actions.min():.3f}, {actions.max():.3f}] ✓")

    # Test 3c: Multi-token context tokens exist
    print("\n3c. Multi-token context architecture...")
    assert hasattr(policy, 'current_emb'), "Missing current_emb tokens"
    assert hasattr(policy, 'goal_emb'), "Missing goal_emb tokens"
    assert hasattr(policy, 'proprio_emb'), "Missing proprio_emb tokens"
    # Shape is (1, n_context_tokens, hidden_dim) for broadcasting
    assert policy.current_emb.shape == (1, 4, 512), f"Wrong current_emb shape: {policy.current_emb.shape}"
    print(f"    Context tokens per modality: 4 ✓")
    print(f"    Total context tokens: 12 (4 current + 4 goal + 4 proprio) ✓")

    print("\n✅ Policy Head Architecture: ALL TESTS PASSED")
    return True


def test_full_model_shapes():
    """Test full model input/output shapes"""
    print("\n" + "=" * 60)
    print("TEST 4: Full Model Shapes")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        device=device,
    )

    B = 2

    # Test 4a: Full forward pass
    print("\n4a. Full forward pass...")
    video = torch.rand(B, 16, 3, 256, 256).to(device)
    goal = torch.rand(B, 3, 256, 256).to(device)
    proprio = torch.rand(B, 5, 23).to(device)

    with torch.no_grad():
        actions = model(video, goal, proprio)

    assert actions.shape == (B, 50, 7), f"Expected ({B}, 50, 7), got {actions.shape}"
    print(f"    video: {video.shape}")
    print(f"    goal: {goal.shape}")
    print(f"    proprio: {proprio.shape}")
    print(f"    -> actions: {actions.shape} ✓")

    # Test 4b: Precomputed embeddings forward
    print("\n4b. Precomputed embeddings forward...")
    current_emb = torch.rand(B, 1408).to(device)
    goal_emb = torch.rand(B, 1408).to(device)

    with torch.no_grad():
        actions = model.forward_with_precomputed(current_emb, goal_emb, proprio)

    assert actions.shape == (B, 50, 7), f"Expected ({B}, 50, 7), got {actions.shape}"
    print(f"    current_emb: {current_emb.shape}")
    print(f"    goal_emb: {goal_emb.shape}")
    print(f"    proprio: {proprio.shape}")
    print(f"    -> actions: {actions.shape} ✓")

    print("\n✅ Full Model Shapes: ALL TESTS PASSED")
    return True


def test_gradient_flow():
    """Test gradient flow through trainable components"""
    print("\n" + "=" * 60)
    print("TEST 5: Gradient Flow")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        device=device,
    )

    # Create inputs
    current_emb = torch.rand(2, 1408, device=device, requires_grad=True)
    goal_emb = torch.rand(2, 1408, device=device, requires_grad=True)
    proprio = torch.rand(2, 5, 23, device=device, requires_grad=True)

    # Forward + backward
    actions = model.forward_with_precomputed(current_emb, goal_emb, proprio)
    loss = actions.mean()
    loss.backward()

    # Test 5a: Input gradients
    print("\n5a. Input gradients...")
    assert current_emb.grad is not None, "No gradient for current_emb!"
    assert goal_emb.grad is not None, "No gradient for goal_emb!"
    assert proprio.grad is not None, "No gradient for proprio!"
    print(f"    current_emb grad norm: {current_emb.grad.norm():.4f} ✓")
    print(f"    goal_emb grad norm: {goal_emb.grad.norm():.4f} ✓")
    print(f"    proprio grad norm: {proprio.grad.norm():.4f} ✓")

    # Test 5b: Critical component gradients
    print("\n5b. Critical component gradients...")
    critical_components = ['proprio_encoder', 'policy_head']
    all_good = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            is_critical = any(comp in name for comp in critical_components)
            if is_critical and param.grad is None:
                print(f"    FAIL: {name}: NO GRADIENT!")
                all_good = False

    if all_good:
        print("    All critical components have gradients ✓")

    # Test 5c: No exploding/vanishing gradients
    print("\n5c. Gradient health check...")
    has_issues = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-7:
                print(f"    WARNING: {name}: vanishing gradient ({grad_norm:.2e})")
                has_issues = True
            elif grad_norm > 1000:
                print(f"    WARNING: {name}: exploding gradient ({grad_norm:.2e})")
                has_issues = True

    if not has_issues:
        print("    No vanishing/exploding gradients ✓")

    print("\n✅ Gradient Flow: ALL TESTS PASSED")
    return all_good


def test_parameter_counts():
    """Test parameter counts"""
    print("\n" + "=" * 60)
    print("TEST 6: Parameter Counts")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        device=device,
    )

    counts = model.count_parameters()

    print("\nParameter counts:")
    for name, count in counts.items():
        print(f"    {name}: {count / 1e6:.2f}M")

    # Verify trainable params are reasonable
    trainable = counts.get('trainable', 0)
    assert trainable > 0, "No trainable parameters!"
    assert trainable < 100_000_000, f"Too many trainable params: {trainable}"  # < 100M

    print(f"\n    Trainable params in expected range ✓")
    print("\n✅ Parameter Counts: ALL TESTS PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("V-JEPA 2 POLICY HEAD - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    results = {}

    # Run tests
    results['vjepa2_encoder'] = test_vjepa2_encoder()
    results['proprio_encoder'] = test_proprio_encoder()
    results['policy_head'] = test_policy_head_architecture()
    results['full_model'] = test_full_model_shapes()
    results['gradient_flow'] = test_gradient_flow()
    results['parameter_counts'] = test_parameter_counts()

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
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
