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
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import stable_pretraining as spt

# Project-specific imports (Absolute relative to sys.path)
from goal_mapper import GoalMapper


def get_img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="gr1_manip")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧬 Initializing GR-1 MPC Evaluator on {device}")

    # 1. Initialize Planning Agent (GoalMapper)
    # This encapsulates the model, memory, and prediction logic
    agent = GoalMapper(cfg.oracle_path, cfg.dataset_root)

    # 2. Set the Planning Target
    success = agent.set_goal(cfg.target_xyz)
    if not success:
        print("❌ Could not find goal frame in dataset. Exiting.")
        return

    print(
        f"🎯 Goal Latent established. Norm: {torch.norm(agent.goal_latent).item():.4f}"
    )

    # 3. Setup World & Solver
    # In a real closed-loop run, we pass the 'agent' directly to the solver
    # as the model, because it now implements the .get_cost() protocol.

    # transform = {"pixels": get_img_transform(cfg)}

    print("✅ Planning Agent ready for MuJoCo closed-loop control.")


if __name__ == "__main__":
    main()
