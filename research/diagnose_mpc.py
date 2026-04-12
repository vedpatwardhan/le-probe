"""
FULL SPECTRUM DIAGNOSTIC SWEEP
Role: Audits the MPC planner across the entire 150-episode distilled gallery.
Mandatory: Requires goal_gallery.pth
"""

import os
import sys
import argparse
import torch
import time
from pathlib import Path

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
        self.action_block = 1


class MockSpace:
    def __init__(self, shape):
        self.shape = shape


def run_diagnostic(model_path, gallery_path="goal_gallery.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔬 Running Full-Spectrum Diagnostic on {device}...")

    if not Path(gallery_path).exists():
        print(f"❌ Error: Gallery not found at {gallery_path}")
        print(
            "💡 Please run 'python research/harvest_goals.py' first to generate the artifact."
        )
        return

    # 1. Load Universal Gallery
    print(f"💎 Loading Gallery: {gallery_path}")
    gallery = torch.load(gallery_path, map_location=device)
    episodes_to_test = sorted(list(gallery["goals"].keys()))
    print(f"📈 Found {len(episodes_to_test)} episodes. Starting full sweep...")

    # 2. Initialize Planning Agent (Gallery Mode)
    # We use a dummy dataset root because the gallery has everything
    agent = GoalMapper(model_path, dataset_root=".")

    # 3. Setup Solver (8k/3.0 Config)
    solver = CEMSolver(
        model=agent, num_samples=8000, var_scale=3.0, n_steps=1, topk=100, device=device
    )
    solver.configure(
        action_space=MockSpace(shape=(1, 64)), n_envs=1, config=MockConfig(horizon=15)
    )

    batch_improvements = []

    # 4. Sequential Audit (All episodes)
    for ep_idx in episodes_to_test:
        print(f"\n🎬 Testing Episode {ep_idx:03d}/{len(episodes_to_test)-1}:")

        # Load Cached Test Case
        agent.goal_latent = gallery["goals"][ep_idx].to(device)
        pixels = gallery["diagnostics"][ep_idx]["pixels"].to(device).unsqueeze(0)
        actions = gallery["diagnostics"][ep_idx]["action"].to(device).unsqueeze(0)

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

    # 5. Global Verdict
    avg_imp = sum(batch_improvements) / len(batch_improvements)
    print(f"\n🏁 FINAL SWEEP VERDICT (n={len(episodes_to_test)}):")
    print(f"   Average Latent Improvement: {avg_imp:.4f}")
    if avg_imp > 50.0:
        print("🚀 VERDICT: MPC Parameters are robust across the entire dataset!")
    else:
        print("⚠️ VERDICT: Weak optimization detected in the global average.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    args = parser.parse_args()
    run_diagnostic(args.model, args.gallery)
