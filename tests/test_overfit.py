"""
CRITICAL: Overfit Test

Verifies the model can memorize a small batch.
If this fails, there's a bug - don't proceed to full training!
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.full_model import VJEPA2Policy


def test_overfit(
    n_samples: int = 10,
    n_epochs: int = 500,
    lr: float = 1e-3,
    target_loss: float = 0.01,
):
    """
    Test that model can overfit to a small batch.

    This is a CRITICAL test. If the model cannot memorize 10 samples,
    there is definitely a bug in the model or training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model with local weights
    print("Creating model...")
    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        vjepa2_freeze=True,
        policy_hidden_dim=512,
        policy_n_layers=4,
        policy_n_context_tokens=4,
        device=device,
    )

    # Create synthetic batch
    print(f"Creating {n_samples} synthetic samples...")

    # Pre-compute embeddings (faster than running V-JEPA 2 each time)
    with torch.no_grad():
        video = torch.rand(n_samples, 16, 3, 256, 256).to(device)
        goal = torch.rand(n_samples, 3, 256, 256).to(device)

        current_emb = model.vjepa2.encode_video(video)
        goal_emb = model.vjepa2.encode_image(goal)

    print(f"  current_emb shape: {current_emb.shape}")  # Should be (n_samples, 1408)
    print(f"  goal_emb shape: {goal_emb.shape}")  # Should be (n_samples, 1408)

    proprio = torch.rand(n_samples, 5, 23).to(device)
    target_actions = torch.rand(n_samples, 50, 7).to(device) * 2 - 1  # [-1, 1]

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

    if final_loss < target_loss:
        print("SUCCESS: OVERFIT TEST PASSED!")
        print("Model can memorize small batch - training should work.")
        return True
    else:
        print("FAILURE: OVERFIT TEST FAILED!")
        print("Model cannot memorize small batch - there's a bug!")
        print("\nDebugging suggestions:")
        print("1. Check gradient flow (test_gradient_flow)")
        print("2. Verify data preprocessing")
        print("3. Check action bounds")
        print("4. Try higher learning rate")
        return False


def test_gradient_flow():
    """Test that gradients flow through the model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        device=device,
    )

    # Create input with correct embedding dimension (1408 for ViT-Giant)
    current_emb = torch.rand(2, 1408, requires_grad=True).to(device)
    goal_emb = torch.rand(2, 1408, requires_grad=True).to(device)
    proprio = torch.rand(2, 5, 23, requires_grad=True).to(device)

    # Forward
    actions = model.forward_with_precomputed(current_emb, goal_emb, proprio)

    # Backward
    loss = actions.mean()
    loss.backward()

    # Check gradients
    print("Gradient check:")

    # Input gradients
    assert current_emb.grad is not None, "No gradient for current_emb!"
    assert goal_emb.grad is not None, "No gradient for goal_emb!"
    assert proprio.grad is not None, "No gradient for proprio!"

    print(f"  current_emb grad norm: {current_emb.grad.norm():.6f}")
    print(f"  goal_emb grad norm: {goal_emb.grad.norm():.6f}")
    print(f"  proprio grad norm: {proprio.grad.norm():.6f}")

    # Parameter gradients
    has_gradient_issue = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"  WARNING: {name}: NO GRADIENT!")
                has_gradient_issue = True
            else:
                grad_norm = param.grad.norm().item()
                if grad_norm < 1e-7:
                    print(f"  WARNING: {name}: grad_norm = {grad_norm:.2e} (vanishing!)")
                    has_gradient_issue = True
                elif grad_norm > 100:
                    print(f"  WARNING: {name}: grad_norm = {grad_norm:.2e} (exploding!)")
                    has_gradient_issue = True

    if not has_gradient_issue:
        print("\nSUCCESS: Gradient flow test passed!")
    else:
        print("\nWARNING: Some gradient issues detected")

    return not has_gradient_issue


def test_shapes():
    """Test that all tensor shapes are correct"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    model = VJEPA2Policy(
        vjepa2_model_path="/workspace/models/vjepa2-ac-vitg.pt",
        vjepa2_model_name="vjepa2_vitg",
        device=device,
    )

    B = 2
    T = 16
    H = W = 256
    proprio_dim = 23
    proprio_history = 5
    action_dim = 7
    chunk_size = 50

    # Test shapes
    video = torch.rand(B, T, 3, H, W).to(device)
    goal = torch.rand(B, 3, H, W).to(device)
    proprio = torch.rand(B, proprio_history, proprio_dim).to(device)

    print(f"Input shapes:")
    print(f"  video: {video.shape}")
    print(f"  goal: {goal.shape}")
    print(f"  proprio: {proprio.shape}")

    with torch.no_grad():
        # Test encoder
        video_emb = model.vjepa2.encode_video(video)
        goal_emb = model.vjepa2.encode_image(goal)
        print(f"\nEncoder outputs:")
        print(f"  video_emb: {video_emb.shape}")  # Should be (B, 1408)
        print(f"  goal_emb: {goal_emb.shape}")    # Should be (B, 1408)

        assert video_emb.shape == (B, 1408), f"Wrong video_emb shape: {video_emb.shape}"
        assert goal_emb.shape == (B, 1408), f"Wrong goal_emb shape: {goal_emb.shape}"

        # Test proprio encoder
        proprio_emb = model.proprio_encoder(proprio)
        print(f"  proprio_emb: {proprio_emb.shape}")  # Should be (B, 256)
        assert proprio_emb.shape == (B, 256), f"Wrong proprio_emb shape: {proprio_emb.shape}"

        # Test full forward
        actions = model(video, goal, proprio)
        print(f"\nOutput:")
        print(f"  actions: {actions.shape}")  # Should be (B, 50, 7)
        assert actions.shape == (B, chunk_size, action_dim), f"Wrong actions shape: {actions.shape}"

        # Check action bounds
        print(f"  action range: [{actions.min():.3f}, {actions.max():.3f}]")
        assert actions.min() >= -1.0 and actions.max() <= 1.0, "Actions out of bounds!"

    print("\nSUCCESS: Shape test passed!")
    return True


if __name__ == "__main__":
    print("="*50)
    print("Running shape test...")
    print("="*50)
    test_shapes()

    print("\n" + "="*50)
    print("Running gradient flow test...")
    print("="*50)
    test_gradient_flow()

    print("\n" + "="*50)
    print("Running overfit test...")
    print("="*50)
    test_overfit()
