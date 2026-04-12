"""
DIAGNOSTIC MPC TUNER (The "Lab Experiment")
Role: Offline validation and hyperparameter tuning of the CEM planner.

This script runs the CEM-MPC logic in a vacuum (no MuJoCo simulation). It is used to:
1. Verify that the CEMSolver + JEPA World Model can successfully reduce latent cost.
2. Tune solver hyperparameters (n_samples, var_scale, horizon) before running simulation.
3. Isolate "Brain" (planning) failures from "Body" (simulation/physics) failures.
"""

import os
import sys
from pathlib import Path

# Add paths for custom modules
CORTEX_GR1 = Path("/Users/vedpatwardhan/Desktop/cortex-os/cortex-gr1")
sys.path.append(str(CORTEX_GR1 / "le_wm"))
sys.path.append(str(CORTEX_GR1 / "research"))

# Setup headless MuJoCo
os.environ["MUJOCO_GL"] = "egl"

# Core imports
import torch
from goal_mapper import GoalMapper
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

    # 1. Initialize Planning Agent
    agent = GoalMapper(MODEL_PATH, ROOT)

    # 2. Set Target (Last frame of success episode)
    success = agent.set_goal(episode_idx=0)
    if not success:
        print("❌ Goal pixels not found.")
        return

    # 3. Solver Setup (Manual Loop for diagnostic)
    solver = CEMSolver(
        model=agent,  # Uses the Agent's .get_cost() and .predict()
        num_samples=1500,
        var_scale=1.2,
        n_steps=1,
        topk=100,
        device=device,
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 64)), n_envs=1, config=MockConfig(horizon=15)
    )

    print("🎬 Generating diagnostic info...")
    pixels = torch.randn(1, 3, 3, 224, 224).to(device)

    # Note: info_dict needs pixels for context
    info_dict = {
        "pixels": pixels,
        "action": torch.zeros(1, 4, 64).to(device),
    }

    print("🎯 Optimizing action sequence (Manual CEM Loops)...")

    current_action = None
    cost_history = []

    for i in range(10):  # 10 high-level iterations
        outputs = solver.solve(info_dict, init_action=current_action)
        current_action = outputs["actions"]
        cost = outputs["costs"][0]
        cost_history.append(cost)
        print(f"  Loop {i:02d}: Average Elite Cost = {cost:.6f}")

    if cost_history[-1] < cost_history[0]:
        print("\n✅ SUCCESS: MPC search successfully reduced latent distance.")
        print(f"   Improvement: {cost_history[0] - cost_history[-1]:.4f}")
    else:
        print("\n⚠️ WARNING: MPC failed to improve.")


if __name__ == "__main__":
    run_diagnostic()
