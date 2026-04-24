"""
ORACLE MPC INFERENCE SERVER (Gallery Edition)
Role: Standalone ZMQ server hosting the JEPA world model and CEM solver.
Mandatory: Requires goal_gallery.pth
"""

import sys
import os
from pathlib import Path

# --- Project Path Stabilization (Aggressive Front-Load) ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LEWM_DIR = os.path.join(ROOT_DIR, "le_wm")
if LEWM_DIR not in sys.path:
    sys.path.insert(0, LEWM_DIR)
# -----------------------------------------------------------

import zmq
import msgpack
import torch
import numpy as np
import time
import argparse
import traceback
import json

# Local imports
from research.goal_mapper import GoalMapper
from stable_worldmodel.solver.cem import CEMSolver
from gr1_protocol import StandardScaler

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 5555


class MockConfig:
    def __init__(self, horizon):
        self.horizon = horizon
        self.action_block = 1


class MockSpace:
    def __init__(self, shape):
        self.shape = shape
        self.low = -1.0
        self.high = 1.0


class LEWMInferenceServer:
    def __init__(self, model_path, gallery_path="goal_gallery.pth"):
        print("--- Initializing Oracle MPC Server (Gallery Only) ---")
        self.scaler = StandardScaler()

        gallery_file = Path(gallery_path)
        if not gallery_file.exists():
            print(f"❌ Error: Gallery not found at {gallery_file}")
            print("💡 Run 'python research/harvest_goals.py' first.")
            exit(1)

        # 1. Load the Universal Gallery
        print(f"💎 Loading Gallery: {gallery_file}")
        self.gallery = torch.load(gallery_file, map_location=DEVICE)
        print(f"✅ Success: {len(self.gallery['goals'])} goal latents ready.")

        # 2. Initialize Brain (Gallery doesn't need data root)
        self.agent = GoalMapper(model_path, dataset_root=".")

        # 3. Load Entire Gallery into Brain (Omni-Goal mode)
        goal_list = [
            self.gallery["goals"][eid] for eid in sorted(self.gallery["goals"].keys())
        ]
        self.agent.goal_latent = torch.stack(goal_list).to(DEVICE)
        print(f"🚀 Brain Prime: Loaded all {len(goal_list)} goals for Omni-MPC.")

        # 4. Setup Calibrated Solver (1k Samples)
        self.solver = CEMSolver(
            model=self.agent,
            num_samples=1000,
            var_scale=1.0,
            n_steps=1,
            topk=100,
            device=DEVICE,
        )
        self.solver.configure(
            action_space=MockSpace(shape=(1, 32)),
            n_envs=1,
            config=MockConfig(horizon=8),
        )

        # 5. State Buffering
        self.history = {"pixels": [], "actions": []}

    def run(self, host="0.0.0.0"):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{PORT}")
        print(f"🚀 LEWM Server LISTENING on port {PORT}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)

                def unpack_np(d):
                    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(
                        d["shape"]
                    )

                raw_image = unpack_np(req.get("world_center"))
                raw_sim_state = unpack_np(req.get("state"))

                # Grounding: Normalize the current state for history alignment
                norm_state = self.scaler.scale_state(raw_sim_state)

                batch = self.agent.transform({"pixels": raw_image})
                image_t = batch["pixels"].to(DEVICE)

                self.history["pixels"].append(image_t)
                if len(self.history["pixels"]) > 3:
                    self.history["pixels"].pop(0)

                # Action History: Ground the model in the current normalized pose
                while len(self.history["actions"]) < 3:
                    self.history["actions"].append(norm_state)

                pixels_stacked = (
                    torch.stack(self.history["pixels"]).unsqueeze(0).unsqueeze(0)
                )
                actions_stacked = (
                    torch.tensor(np.stack(self.history["actions"]), dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                print("🧠 Step: Planning (8,000 parallel samples)...")
                start_time = time.time()
                with torch.inference_mode():
                    # 🚀 WARM-START CEM: Pass previous action as initial guess
                    last_executed_action = actions_stacked[
                        :, :, -1:, :
                    ]  # (1, 1, 1, 32)
                    init_guess = last_executed_action.expand(
                        -1, -1, 8, -1
                    )  # Repeat for horizon 8

                    outputs = self.solver.solve(
                        {"pixels": pixels_stacked, "action": actions_stacked},
                        init_action=init_guess,
                    )

                best_plan = outputs["actions"].cpu().numpy()
                target_action = best_plan[0, 0, 0]  # (32,)

                # 🛡️ THE GOVERNOR: Delta Capping & Smoothing 🛡️
                if len(self.history["actions"]) > 0:
                    prev_action = self.history["actions"][
                        -1
                    ]  # This is already a numpy array

                    # 1. Delta Capping (Max 0.05 normalized units per step)
                    max_delta = 0.05
                    delta = target_action - prev_action
                    clipped_delta = np.clip(delta, -max_delta, max_delta)
                    target_action = prev_action + clipped_delta

                    # 2. Action Smoothing (Exponential Moving Average)
                    # Blends 90% new intent with 10% previous pose to kill high-frequency jitter
                    alpha = 0.9
                    target_action = alpha * target_action + (1 - alpha) * prev_action

                # Update the plan with our smoothed/clipped action for execution
                if best_plan.ndim == 4:
                    best_plan = best_plan[0, 0]  # (B, S, T, D) -> (T, D)
                elif best_plan.ndim == 3:
                    best_plan = best_plan[0]  # (S, T, D) -> (T, D)

                best_plan[0] = target_action

                plan_time = time.time() - start_time

                # Diagnostic Logging: Action Stats
                print(
                    f"🧠 Planning Stats -> Solve Time: {plan_time:.2f}s, "
                    f"Max Action: {np.abs(best_plan).max():.4f}, "
                    f"Mean Action: {np.abs(best_plan).mean():.4f}"
                )

                # --- 📡 Rerun Telemetry & Audit ---
                self.log_diagnostics(
                    raw_image=raw_image,
                    best_plan=best_plan,
                    plan_time=plan_time,
                    instruction=req.get("instruction", "Unknown"),
                )

                # Update history with the first action of the plan (which is normalized)
                self.history["actions"].append(best_plan[0])
                if len(self.history["actions"]) > 3:
                    self.history["actions"].pop(0)

                socket.send(
                    msgpack.packb(
                        {
                            "action": best_plan.tolist(),
                            "diagnostics": {"plan_time_ms": int(plan_time * 1000)},
                        },
                        use_bin_type=True,
                    )
                )

            except Exception as e:
                print(f"❌ Server Error: {e}")
                traceback.print_exc()
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))

    def log_diagnostics(self, raw_image, best_plan, plan_time, instruction):
        """Pure JSONL logging for 'Wild Movement' debugging."""
        try:
            # Lifecycle Audit (JSONL)
            log_entry = {
                "timestamp": time.time(),
                "instruction": instruction,
                "solve_time": plan_time,
                "action_max": float(np.abs(best_plan).max()),
                "action_mean": float(np.abs(best_plan).mean()),
                "first_action_norm": best_plan[0].tolist(),
                "first_action_raw": self.scaler.unscale_action(best_plan[0]).tolist(),
            }
            log_file = os.path.join(ROOT_DIR, "lewm_lifecycle_audit.json")
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            print(f"⚠️ Diagnostic logging failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    args = parser.parse_args()
    server = LEWMInferenceServer(args.model, args.gallery)
    server.run()
