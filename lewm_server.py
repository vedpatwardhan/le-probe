"""
ORACLE MPC INFERENCE SERVER (Gallery Edition)
Role: Standalone ZMQ server hosting the JEPA world model and CEM solver.
Mandatory: Requires goal_gallery.pth
"""

import zmq
import msgpack
import torch
import numpy as np
import time
import argparse
from pathlib import Path

# Local imports
from research.goal_mapper import GoalMapper
from stable_worldmodel.solver.cem import CEMSolver

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


class LEWMInferenceServer:
    def __init__(self, model_path, gallery_path="goal_gallery.pth"):
        print("--- Initializing Oracle MPC Server (Gallery Only) ---")

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

        # 4. Setup Calibrated Solver (8k Samples)
        self.solver = CEMSolver(
            model=self.agent,
            num_samples=8000,
            var_scale=3.0,
            n_steps=1,
            topk=100,
            device=DEVICE,
        )
        self.solver.configure(
            action_space=MockSpace(shape=(1, 64)),
            n_envs=1,
            config=MockConfig(horizon=15),
        )

        # 5. State Buffering
        self.history = {"pixels": [], "actions": []}

    def map_sim_to_model_state(self, state_32):
        state_full = np.zeros(64, dtype=np.float32)
        state_full[0:7] = state_32[0:7]
        state_full[7:13] = state_32[7:13]
        state_full[19:22] = state_32[13:16]
        state_full[22:29] = state_32[16:23]
        state_full[29:35] = state_32[23:29]
        state_full[41:44] = state_32[29:32]
        return state_full

    def map_model_to_sim_actions(self, actions_64):
        horizon = actions_64.shape[0]
        actions_32 = np.zeros((horizon, 32), dtype=np.float32)
        actions_32[:, 0:7] = actions_64[:, 0:7]
        actions_32[:, 7:13] = actions_64[:, 7:13]
        actions_32[:, 13:16] = actions_64[:, 19:22]
        actions_32[:, 16:23] = actions_64[:, 22:29]
        actions_32[:, 23:29] = actions_64[:, 29:35]
        actions_32[:, 29:32] = actions_64[:, 41:44]
        return actions_32

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
                sim_state = unpack_np(req.get("state"))

                batch = self.agent.transform({"pixels": raw_image})
                image_t = batch["pixels"].to(DEVICE)

                self.history["pixels"].append(image_t)
                if len(self.history["pixels"]) > 3:
                    self.history["pixels"].pop(0)

                while len(self.history["actions"]) < 3:
                    self.history["actions"].append(np.zeros(64, dtype=np.float32))

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
                    planned_actions = self.solver.solve(
                        {"pixels": pixels_stacked, "action": actions_stacked}
                    )

                best_plan_64 = planned_actions[0].cpu().numpy()
                plan_time = time.time() - start_time

                best_plan_32 = self.map_model_to_sim_actions(best_plan_64)

                self.history["actions"].append(best_plan_64[0])
                if len(self.history["actions"]) > 3:
                    self.history["actions"].pop(0)

                socket.send(
                    msgpack.packb(
                        {
                            "action": best_plan_32.tolist(),
                            "diagnostics": {"plan_time_ms": int(plan_time * 1000)},
                        },
                        use_bin_type=True,
                    )
                )

            except Exception as e:
                print(f"❌ Server Error: {e}")
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    args = parser.parse_args()
    server = LEWMInferenceServer(args.model, args.gallery)
    server.run()
