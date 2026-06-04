# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import numpy as np
import time
import gc
import mujoco
import argparse
import json
from tqdm import tqdm
from PIL import Image

from simulation_base import GR1MuJoCoBase
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler
from dataset.lerobot_manager import LeRobotManager


class AutoDatasetGenerator(GR1MuJoCoBase):
    """
    Automated dataset generation pipeline utilizing the IK solver.
    Allows headless generation of 2000 episodes split across:
      - Successful episodes (500)
      - Sub-optimal/Perturbed episodes (1000)
      - Failing/Horrible episodes (500)
    """

    def __init__(self, dataset_name="gr1_auto_dataset", seed=42, recorder=None):
        super().__init__(scene_path=SCENE_PATH, restrict_ik=True)
        np.random.seed(seed)

        # Override the recorder to save to the custom target dataset
        if recorder is not None:
            self.recorder = recorder
        else:
            self.recorder = LeRobotManager(
                repo_id=dataset_name, fps=10, upload_interval=50
            )
        print(f"📁 Initialized dataset generator for: {dataset_name}")

    def run_automatic_generation(
        self, total_episodes=2000, success_ratio=0.25, suboptimal_ratio=0.50
    ):
        n_success = int(total_episodes * success_ratio)
        n_suboptimal = int(total_episodes * suboptimal_ratio)
        n_fail = total_episodes - n_success - n_suboptimal

        print(f"\n🚀 Starting dataset generation sequence:")
        print(f"  - Successful episodes: {n_success}")
        print(f"  - Sub-optimal episodes: {n_suboptimal}")
        print(f"  - Failing episodes: {n_fail}\n")

        episode_types = (
            ["success"] * n_success + ["suboptimal"] * n_suboptimal + ["fail"] * n_fail
        )
        np.random.shuffle(episode_types)  # Interleave for data variety

        for ep_idx, ep_type in enumerate(
            tqdm(episode_types, desc="Generating episodes")
        ):
            # Instantiate a fresh generator for each episode to completely reset MuJoCo, Mink, and rendering state
            # but reuse self.recorder to avoid expensive dataset reinitialization
            ep_generator = AutoDatasetGenerator(
                dataset_name=self.recorder.repo_id, seed=ep_idx, recorder=self.recorder
            )
            try:
                ep_generator.generate_episode(ep_idx, ep_type)
            finally:
                ep_generator.close()
                del ep_generator
                gc.collect()

        # Flush any remaining cached episodes that didn't fill the final batch
        self.recorder.flush_accumulated_episodes()

        print("\n🎉 Automated generation sequence completed successfully!")

    def generate_episode(self, ep_idx, ep_type):
        """Runs the 4-phase sequence with type-specific perturbations."""
        t_start_ep = time.time()

        # 1. Reset Env (Randomizes cube within table boundaries)
        t_reset_start = time.time()
        self.reset_env(lock_posture=True, randomize_cube=True)
        print(
            f"[TIMER] generate_episode: reset_env took {time.time() - t_reset_start:.4f}s"
        )

        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_pos = self.data.qpos[
            self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id] + 3
        ].copy()

        # Start recording the episode
        task_desc = f"Pick up red cube (Class: {ep_type})"
        t_start_rec = time.time()
        self.recorder.start_episode(task_desc)
        print(
            f"[TIMER] generate_episode: start_episode took {time.time() - t_start_rec:.4f}s"
        )
        self.is_recording = True

        # Define offsets and parameter mutations based on episode type
        offsets = {
            "success": 0.0,
            "suboptimal": np.random.uniform(0.07, 0.3),  # 7 - 30 cm perturbation
            "fail": np.random.uniform(0.3, 0.7),  # 30 - 70 cm massive failure
        }

        offset = offsets[ep_type]
        quat_down = [0, 1, 0, 0]

        try:
            # ==========================================
            # PHASE 1: Approach / Rotate
            # ==========================================
            t_phase1_start = time.time()
            self.current_phase = 1
            # Add random directional noise to the target positions for suboptimal/failed runs
            noise_p1 = (
                np.random.normal(0, offset, size=3)
                if ep_type in ["suboptimal", "fail"]
                else np.zeros(3)
            )
            pos_i_h = cube_pos + [0.02, 0.02, 0.07] + noise_p1
            pos_t_h = cube_pos + [-0.02, 0, 0.07] + noise_p1
            pos_w_h = cube_pos + [0, 0, 0.13] + noise_p1

            t_ik_start = time.time()
            q_reach_h = self.solve_ik(
                pos_w_h, quat_down, pos_i_h, pos_t_h, posture_cost=1e-6
            )
            t_ik_dur = time.time() - t_ik_start

            t_dispatch_start = time.time()
            self.dispatch_action(
                self.qpos_to_action_32(q_reach_h),
                q_reach_h,
                n_steps=240,
                render_freq=30,
            )
            t_dispatch_dur = time.time() - t_dispatch_start
            print(
                f"[TIMER] generate_episode: Phase 1 took {time.time() - t_phase1_start:.4f}s (IK: {t_ik_dur:.4f}s, Dispatch: {t_dispatch_dur:.4f}s)"
            )

            # ==========================================
            # PHASE 2: Descent
            # ==========================================
            t_phase2_start = time.time()
            self.current_phase = 2
            noise_p2 = (
                np.random.normal(0, offset, size=3)
                if ep_type in ["suboptimal", "fail"]
                else np.zeros(3)
            )
            pos_i_l = cube_pos + [-0.02, 0.02, 0] + noise_p2
            pos_t_l = cube_pos + [-0.06, 0, 0] + noise_p2
            pos_w_l = cube_pos + [0, 0, 0.06] + noise_p2

            t_ik_start = time.time()
            q_reach_l = self.solve_ik(
                pos_w_l, quat_down, pos_i_l, pos_t_l, posture_cost=1e-6
            )
            t_ik_dur = time.time() - t_ik_start

            # Keep fingers open
            for f_idx in [50, 51, 52, 53, 54, 55, 56]:
                if f_idx < len(q_reach_l):
                    q_reach_l[f_idx] = 0.0

            t_dispatch_start = time.time()
            self.dispatch_action(
                self.qpos_to_action_32(q_reach_l),
                q_reach_l,
                n_steps=240,
                render_freq=30,
            )
            t_dispatch_dur = time.time() - t_dispatch_start
            print(
                f"[TIMER] generate_episode: Phase 2 took {time.time() - t_phase2_start:.4f}s (IK: {t_ik_dur:.4f}s, Dispatch: {t_dispatch_dur:.4f}s)"
            )

            # ==========================================
            # PHASE 3: Grasp
            # ==========================================
            t_phase3_start = time.time()
            self.current_phase = 3
            noise_p3 = (
                np.random.normal(0, offset * 0.5, size=3)
                if ep_type == "suboptimal"
                else np.zeros(3)
            )
            # Fails completely by placing the grasp target far away
            if ep_type == "fail":
                noise_p3 = np.random.normal(0, 0.12, size=3)

            pos_i_l = cube_pos + [0, 0.02, 0] + noise_p3
            pos_t_l = cube_pos + [0, 0, 0] + noise_p3
            pos_w_l = cube_pos + [0, 0, 0] + noise_p3

            t_ik_start = time.time()
            q_reach_l = self.solve_ik(
                pos_w_l, quat_down, pos_i_l, pos_t_l, posture_cost=1e-6
            )
            t_ik_dur = time.time() - t_ik_start
            q_grasp = q_reach_l.copy()

            # If failing, make the grip loose or wrong joint command limits
            grip_force = 0.3 if ep_type == "fail" else 1.1
            q_grasp[48] = grip_force
            for g_id in [50, 52, 54, 56]:
                q_grasp[g_id] = -grip_force

            t_dispatch_start = time.time()
            self.dispatch_action(
                self.qpos_to_action_32(q_grasp), q_grasp, n_steps=240, render_freq=30
            )
            t_dispatch_dur = time.time() - t_dispatch_start
            print(
                f"[TIMER] generate_episode: Phase 3 took {time.time() - t_phase3_start:.4f}s (IK: {t_ik_dur:.4f}s, Dispatch: {t_dispatch_dur:.4f}s)"
            )

            # ==========================================
            # PHASE 4: Lift / Retract
            # ==========================================
            t_phase4_start = time.time()
            self.current_phase = 4
            noise_p4 = (
                np.random.normal(0, offset, size=3)
                if ep_type in ["suboptimal", "fail"]
                else np.zeros(3)
            )
            pos_i_up = cube_pos + [0, 0.02, 0.25] + noise_p4
            pos_t_up = cube_pos + [0, 0, 0.25] + noise_p4
            pos_w_up = cube_pos + [0, 0, 0.25] + noise_p4

            t_ik_start = time.time()
            q_lift = self.solve_ik(
                pos_w_up, quat_down, pos_i_up, pos_t_up, posture_cost=1e-6
            )
            t_ik_dur = time.time() - t_ik_start

            # Maintain grip force (or drop the cube for failed attempts)
            lift_grip_force = 0.0 if ep_type == "fail" else grip_force
            q_lift[48] = lift_grip_force
            for g_id in [50, 52, 54, 56]:
                q_lift[g_id] = -lift_grip_force

            t_dispatch_start = time.time()
            self.dispatch_action(
                self.qpos_to_action_32(q_lift), q_lift, n_steps=240, render_freq=30
            )
            t_dispatch_dur = time.time() - t_dispatch_start
            print(
                f"[TIMER] generate_episode: Phase 4 took {time.time() - t_phase4_start:.4f}s (IK: {t_ik_dur:.4f}s, Dispatch: {t_dispatch_dur:.4f}s)"
            )

            # 2. Stop and save episode
            t_stop_rec_start = time.time()
            self.recorder.stop_episode()
            print(
                f"[TIMER] generate_episode: stop_episode took {time.time() - t_stop_rec_start:.4f}s"
            )
            print(
                f"[TIMER] generate_episode: TOTAL episode {ep_idx} took {time.time() - t_start_ep:.4f}s\n"
            )
        except Exception as e:
            print(f"⚠️ Error generating episode {ep_idx}: {str(e)}")
            self.recorder.discard_episode()
        finally:
            self.is_recording = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated GR-1 Simulation Dataset Generator"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="gr1_pickup_2k_hybrid",
        help="LeRobot output repo_id",
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Total episodes to generate"
    )
    parser.add_argument(
        "--success-ratio",
        type=float,
        default=0.25,
        help="Ratio of successful episodes (500 out of 2000)",
    )
    parser.add_argument(
        "--suboptimal-ratio",
        type=float,
        default=0.50,
        help="Ratio of sub-optimal episodes (1000 out of 2000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = AutoDatasetGenerator(dataset_name=args.dataset_name, seed=args.seed)
    generator.run_automatic_generation(
        total_episodes=args.episodes,
        success_ratio=args.success_ratio,
        suboptimal_ratio=args.suboptimal_ratio,
    )
