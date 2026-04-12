"""
MPC EVALUATION COMMANDER (The "Mission")
Role: Closed-loop MuJoCo simulation and benchmarking.

This is the primary execution script for the GR-1 MPC pipeline. It:
1. Connects the Planning Agent (GoalMapper) to the MuJoCo simulator (GR1MuJoCoBase).
2. Runs the 20Hz control loop: Perception -> CEM Planning -> Physical Execution.
3. Tracks success metrics for Reach, Grasp, and Lift tasks.
"""

import os
import sys
import time
import torch
import hydra
from pathlib import Path
from collections import deque
from omegaconf import DictConfig
from torchvision.transforms import v2 as transforms

# Project paths
CORTEX_GR1 = Path("/Users/vedpatwardhan/Desktop/cortex-os/cortex-gr1")
sys.path.append(str(CORTEX_GR1 / "le_wm"))
sys.path.append(str(CORTEX_GR1))  # For simulation_base

# Setup headless MuJoCo
os.environ["MUJOCO_GL"] = "egl"

# Project-specific imports
import stable_pretraining as spt
from simulation_base import GR1MuJoCoBase
from research.goal_mapper import GoalMapper
from stable_worldmodel.solver import CEMSolver


class GR1EvalCommander(GR1MuJoCoBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Initialize Planning Agent
        self.agent = GoalMapper(cfg.oracle_path, cfg.dataset_root)
        self.agent.set_goal(episode_idx=0)

        # 2. Initialize Solver (The Tuned "Thinking" engine)
        class MockConfig:  # For solver compatibility
            def __init__(self, h):
                self.horizon = h

        class MockSpace:
            def __init__(self, s):
                self.shape = s

        self.solver = CEMSolver(
            model=self.agent,
            num_samples=cfg.eval.solver.num_samples,
            var_scale=cfg.eval.solver.var_scale,
            n_steps=cfg.eval.solver.n_steps,
            topk=cfg.eval.solver.topk,
            device=self.device,
        )
        self.solver.configure(
            action_space=MockSpace(shape=(1, 64)),
            n_envs=1,
            config=MockConfig(h=cfg.eval.solver.horizon),
        )

        # 3. Perception Buffer (JEPA needs T=3 history)
        self.history = deque(maxlen=3)
        self.img_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=(224, 224)),
            ]
        )

    def get_observation(self):
        """Captures and transforms the world_center camera view."""
        self.renderer.update_scene(self.data, camera="world_center")
        rgb = self.renderer.render()
        pixels = (
            self.img_transform(rgb).to(self.device).unsqueeze(0)
        )  # (1, 3, 224, 224)
        return pixels

    def run_mission(self, max_steps=50):
        print(
            f"🚀 MISSION START: Goal Latent established (Norm: {torch.norm(self.agent.goal_latent).item():.4f})"
        )

        # Warmup history buffer
        print("📥 Filling perception history...")
        for _ in range(3):
            self.history.append(self.get_observation())
            time.sleep(0.05)

        current_action_seq = None

        for step in range(max_steps):
            # 1. Perception
            # Combine history into (1, T, C, H, W)
            pixels_batch = torch.stack(list(self.history), dim=1)  # (1, 3, 3, 224, 224)
            current_state = self.get_state_32()

            info_dict = {
                "pixels": pixels_batch,
                "action": torch.zeros(1, 4, 32).to(
                    self.device
                ),  # Action history (dummy for now)
            }

            # 2. Planning (CEM)
            print(f"🧠 Step {step:02d}: Planning move...")
            outputs = self.solver.solve(info_dict, init_action=current_action_seq)

            # Receding Horizon: Take only the first action of the imagined sequence
            best_action_chunk = outputs[
                "actions"
            ]  # (B, Horizon, 64) -> We take (1, 1, 32)?
            # Note: Solver expects 64-dim in config, but we map to 32 joints.
            # We'll take the first 32 dims of the first step.
            action_32 = best_action_chunk[0, 0, :32].cpu().numpy()

            # 3. Execution (The "Body")
            self.process_target_32(action_32)
            # Use IK to reach the target configuration
            # In this simple protocol, process_target_32 updates last_target_q
            self.dispatch_action(action_32, self.last_target_q)

            # 4. Update perception for transition
            self.history.append(self.get_observation())

        print("🏁 Mission Complete.")


@hydra.main(version_base=None, config_path="./config/eval", config_name="gr1_manip")
def main(cfg: DictConfig):
    commander = GR1EvalCommander(cfg)
    commander.run_mission()


if __name__ == "__main__":
    main()
