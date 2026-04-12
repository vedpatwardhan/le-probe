"""
DIAGNOSTIC MPC TUNER (The "Lab Experiment")
Role: Offline validation and hyperparameter tuning of the CEM planner.
"""

import os
import sys
import argparse
import torch
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# Project paths
RESEARCH_DIR = Path(__file__).parent.absolute()
CORTEX_GR1 = RESEARCH_DIR.parent
sys.path.append(str(CORTEX_GR1))
sys.path.append(str(CORTEX_GR1 / "le_wm"))

from goal_mapper import GoalMapper
from stable_worldmodel.solver import CEMSolver


class MockConfig:
    def __init__(self, horizon):
        self.horizon = horizon


class MockSpace:
    def __init__(self, shape):
        self.shape = shape


def run_diagnostic(model_path, gallery_path="goal_gallery.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running MPC Latent Diagnostic on {device}...")

    gallery = None
    if Path(gallery_path).exists():
        print(f"💎 Loading Diagnostic Gallery from: {gallery_path}")
        gallery = torch.load(gallery_path, map_location=device)
        dataset_root = "."  # Dummy
    else:
        print(f"☁️ Gallery not found. Fallback: Syncing dataset from Hub...")
        dataset_root = snapshot_download(
            repo_id="vedpatwardhan/gr1_pickup_processed", repo_type="dataset"
        )

    # 1. Initialize Planning Agent
    agent = GoalMapper(model_path, dataset_root)

    # 2. Setup Solver
    solver = CEMSolver(
        model=agent, num_samples=8000, var_scale=3.0, n_steps=1, topk=100, device=device
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 64)), n_envs=1, config=MockConfig(horizon=15)
    )

    # 3. Batch Tuning Loop
    episodes_to_test = [0, 30, 60, 90, 120, 149]
    batch_improvements = []

    for ep_idx in episodes_to_test:
        print(f"\n🎬 Testing Episode {ep_idx:03d}:")

        # Pull Goal & Start Frames
        if gallery and ep_idx in gallery["goals"]:
            agent.goal_latent = gallery["goals"][ep_idx].to(device)
            pixels = (
                gallery["diagnostics"][ep_idx]["pixels"].to(device).unsqueeze(0)
            )  # (1, 3, 3, 224, 224)
            actions = (
                gallery["diagnostics"][ep_idx]["action"].to(device).unsqueeze(0)
            )  # (1, 4, 64)
            print("   ✅ Loaded Goal & Start Frames from Gallery")
        else:
            success = agent.set_goal(episode_idx=ep_idx)
            if not success:
                continue
            pixels = torch.randn(1, 3, 3, 224, 224).to(device)
            actions = torch.zeros(1, 4, 64).to(device)
            print("   ⚠️ No cached test case. Using noise start.")

        info_dict = {"pixels": pixels, "action": actions}
        current_action = None
        start_cost, end_cost = None, None

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
    if batch_improvements:
        avg_imp = sum(batch_improvements) / len(batch_improvements)
        print(f"\n🏁 BATCH VERDICT:\n   Average Latent Improvement: {avg_imp:.4f}")
        if avg_imp > 50.0:
            print("🚀 VERDICT: MPC Parameters are robust and ready for simulator!")
        else:
            print("⚠️ VERDICT: Optimization is weak. Consider increasing num_samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    args = parser.parse_args()
    run_diagnostic(args.model, args.gallery)
