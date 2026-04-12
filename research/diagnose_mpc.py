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
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

# Project Constants
REPO_ID = "vedpatwardhan/gr1_pickup_processed"

# Resolve project paths dynamically
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
sys.path.append(str(CORTEX_GR1))
sys.path.append(str(CORTEX_GR1 / "le_wm"))

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


def run_diagnostic(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running Batch MPC Latent Diagnostic on {device}...")

    # Mandatory Dataset Download from Hub
    print(f"☁️ Synchronizing dataset from Hugging Face Hub: {REPO_ID}...")
    dataset_root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    print(f"✅ Dataset ready at: {dataset_root}")
    print(f"📁 Weights: {model_path}")

    # 1. Initialize Planning Agent
    agent = GoalMapper(model_path, dataset_root)

    # 2. Setup Solver
    solver = CEMSolver(
        model=agent,
        num_samples=8000,
        var_scale=3.0,
        n_steps=1,
        topk=100,
        device=device,
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 64)), n_envs=1, config=MockConfig(horizon=15)
    )

    # 3. Batch Tuning Loop
    episodes_to_test = [0, 30, 60, 90, 120, 149]
    batch_improvements = []

    print(
        f"📊 Auditing planning performance across {len(episodes_to_test)} episodes..."
    )

    for ep_idx in episodes_to_test:
        print(f"\n🎬 Testing Episode {ep_idx:03d}:")
        success = agent.set_goal(episode_idx=ep_idx)
        if not success:
            continue

        pixels = torch.randn(1, 3, 3, 224, 224).to(device)
        info_dict = {"pixels": pixels, "action": torch.zeros(1, 4, 64).to(device)}

        current_action = None
        start_cost = None
        end_cost = None

        for i in range(10):
            outputs = solver.solve(info_dict, init_action=current_action)
            current_action = outputs["actions"]
            cost = outputs["costs"][0]
            if i == 0:
                start_cost = cost
            end_cost = cost
            print(f"  Loop {i}: Cost = {cost:.4f}")

        improvement = start_cost - end_cost
        batch_improvements.append(improvement)
        print(f"✅ Ep {ep_idx} Improvement: {improvement:.4f}")

    # 4. Final Verdict
    if not batch_improvements:
        print("❌ Error: No episodes were successfully tested.")
        return

    avg_imp = sum(batch_improvements) / len(batch_improvements)
    print(f"\n🏁 BATCH VERDICT:")
    print(f"   Episodes Tested: {len(episodes_to_test)}")
    print(f"   Average Latent Improvement: {avg_imp:.4f}")

    if avg_imp > 50:
        print("🚀 VERDICT: MPC Parameters are robust and ready for simulator!")
    else:
        print(
            "⚠️ VERDICT: Optimization is weak. Consider increasing num_samples or var_scale."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR-1 MPC Diagnostic Tuner")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/vedpatwardhan/Desktop/cortex-os/lewm_baseline/outputs/gr1_prod_v17/checkpoints/gr1-epoch=99-step=005400.ckpt",
        help="Path to the JEPA Oracle checkpoint",
    )
    args = parser.parse_args()

    run_diagnostic(args.model_path)
