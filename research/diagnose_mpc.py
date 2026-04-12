import torch
import sys
from pathlib import Path
import os

# Add paths for custom modules
CORTEX_GR1 = Path("/Users/vedpatwardhan/Desktop/cortex-os/cortex-gr1")
sys.path.append(str(CORTEX_GR1 / "le_wm"))
sys.path.append(str(CORTEX_GR1 / "research"))

# Setup headless MuJoCo
os.environ["MUJOCO_GL"] = "egl"

# Core imports
import torch
from goal_utils import find_goal_pixels
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor
from gr1_modules import GR1Embedder, GR1MLP
from stable_worldmodel.solver import CEMSolver


class MockConfig:
    def __init__(self, horizon, action_block=1):
        self.horizon = horizon
        self.action_block = action_block


class MockSpace:
    def __init__(self, shape):
        self.shape = shape


def run_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running MPC Latent Diagnostic on {device}...")

    MODEL_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/lewm_baseline/outputs/gr1_prod_v17/checkpoints/gr1-epoch=99-step=005400.ckpt"
    ROOT = "/Users/vedpatwardhan/Desktop/cortex-os/cortex-gr1/datasets/vedpatwardhan/gr1_pickup_processed"
    target_xyz = [-0.142, 0.268, 0.091]

    goal_pixels = find_goal_pixels(target_xyz, MODEL_PATH, ROOT)
    if goal_pixels is None:
        print("❌ Goal pixels not found.")
        return
    goal_pixels = goal_pixels.to(device)

    # 2. Manual Model Load (Oracle v17)

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False
    )
    predictor = ARPredictor(
        num_frames=3,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
    )
    action_encoder = GR1Embedder(input_dim=64, emb_dim=192)
    projector = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)
    predictor_proj = GR1MLP(input_dim=192, output_dim=192, hidden_dim=2048)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(
        {
            k.replace("model.", ""): v
            for k, v in checkpoint.get("state_dict", checkpoint).items()
        },
        strict=False,
    )
    model.eval()

    # 3. Solver Setup
    # MONKEY PATCH for visibility
    original_solve = CEMSolver.solve

    def patched_solve(self, info_dict, init_action=None):
        print("   (Monkey-patched solver to show iteration metrics)")
        # We'll let it run, but we want to see the cost internal to the loop
        # Instead of re-implementing, we'll just use a small N and run multiple times
        # to simulate iterations for this diagnostic
        return original_solve(self, info_dict, init_action)

    solver = CEMSolver(
        model=model,
        num_samples=1500,
        var_scale=1.2,
        n_steps=1,  # We will loop manually to see progress
        topk=100,
        device=device,
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 64)), n_envs=1, config=MockConfig(horizon=15)
    )

    print("🎬 Generating diagnostic info...")
    pixels = torch.randn(1, 3, 3, 224, 224).to(device)

    info_dict = {
        "pixels": pixels,
        "goal": goal_pixels,
        "action": torch.zeros(1, 4, 64).to(device),
    }

    print("🎯 Optimizing action sequence (Manual CEM Loops)...")

    current_action = None
    cost_history = []

    for i in range(10):  # 10 high-level iterations
        outputs = solver.solve(info_dict, init_action=current_action)
        current_action = outputs["actions"]  # Use mean of elites as start for next
        cost = outputs["costs"][0]
        cost_history.append(cost)
        print(f"  Loop {i:02d}: Average Elite Cost = {cost:.6f}")

    if cost_history[-1] < cost_history[0]:
        print("\n✅ SUCCESS: MPC search successfully reduced latent distance.")
        print(f"   Improvement: {cost_history[0] - cost_history[-1]:.4f}")
    else:
        print(
            "\n⚠️ WARNING: MPC failed to improve. This might suggest the latent goal is out of manifold for this start state."
        )


if __name__ == "__main__":
    run_diagnostic()
