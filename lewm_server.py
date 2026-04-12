"""
ORACLE MPC INFERENCE SERVER
Role: Standalone ZMQ server hosting the JEPA world model and CEM solver.

Key Features:
1. Internal History Buffering (maintains the 3-frame sliding window).
2. Rosetta Mapping (32-dim simulator joints <-> 64-dim dataset manifold).
3. Calibrated Search (8k samples, 3.0 variance, 10x cost pressure).
4. Goal Gallery Support (Uses 115KB cache instead of 100GB dataset).
"""

import zmq
import msgpack
import torch
import numpy as np
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

# Local imports from the research layer
from research.goal_mapper import GoalMapper
from stable_worldmodel.solver.cem import CEMSolver

# Configuration Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 5555
REPO_ID = "vedpatwardhan/gr1_pickup_processed"
DEFAULT_GALLERY = "goal_gallery.pth"


class LEWMInferenceServer:
    def __init__(self, model_path, dataset_root=None, gallery_path=DEFAULT_GALLERY):
        print("--- Initializing Oracle MPC Server ---")

        self.gallery = None
        gallery_file = Path(gallery_path)

        # 0. Load Goal Gallery (The Optimized Path 🚀)
        if gallery_file.exists():
            print(f"💎 Loading Goal Gallery from: {gallery_file}")
            self.gallery = torch.load(gallery_file, map_location=DEVICE)
            print(f"✅ Success: {len(self.gallery)} goal latents cached.")

            # If we have a gallery, we don't need the dataset root!
            if dataset_root is None:
                dataset_root = "."  # Dummy path to satisfy GoalMapper init
        else:
            # 0b. Sync Dataset only if gallery is missing (The Legacy Path ☁️)
            if dataset_root is None or not Path(dataset_root).exists():
                print(
                    f"⚠️ Gallery not found. Fallback: Syncing dataset from Hub: {REPO_ID}..."
                )
                dataset_root = snapshot_download(repo_id=REPO_ID, repo_type="dataset")

        # 1. Initialize the Tuned Brain
        self.agent = GoalMapper(model_path, dataset_root)

        # 2. Initialize the Calibrated Solver
        self.solver = CEMSolver(
            model=self.agent,
            num_samples=8000,
            var_scale=3.0,
            n_steps=1,
            topk=100,
            device=DEVICE,
        )
        self.solver.configure(horizon=15)

        # 3. State/History Buffering
        self.history = {
            "pixels": [],
            "actions": [],
        }
        self.goal_set = False

    def map_sim_to_model_state(self, state_32):
        """Translates 32-dim sim joints to 64-dim manifold."""
        state_full = np.zeros(64, dtype=np.float32)
        state_full[0:7] = state_32[0:7]  # Left Arm
        state_full[7:13] = state_32[7:13]  # Left Hand
        state_full[19:22] = state_32[13:16]  # Neck/Head
        state_full[22:29] = state_32[16:23]  # Right Arm
        state_full[29:35] = state_32[23:29]  # Right Hand
        state_full[41:44] = state_32[29:32]  # Waist
        return state_full

    def map_model_to_sim_actions(self, actions_64):
        """Translates 64-dim manifold actions back to 32-dim sim joints."""
        horizon = actions_64.shape[0]
        actions_32 = np.zeros((horizon, 32), dtype=np.float32)

        actions_32[:, 0:7] = actions_64[:, 0:7]  # Left Arm
        actions_32[:, 7:13] = actions_64[:, 7:13]  # Left Hand
        actions_32[:, 13:16] = actions_64[:, 19:22]  # Neck/Head
        actions_32[:, 16:23] = actions_64[:, 22:29]  # Right Arm
        actions_32[:, 23:29] = actions_64[:, 29:35]  # Right Hand
        actions_32[:, 29:32] = actions_64[:, 41:44]  # Waist
        return actions_32

    def run(self, host="0.0.0.0"):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{PORT}")
        print(f"🚀 LEWM MPC Server listening on port {PORT}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)

                # Goal Setting Logic
                if not self.goal_set:
                    if self.gallery is not None:
                        print(
                            "🎯 Mission Start: Loading goal latent from Gallery [Ep 000]..."
                        )
                        self.agent.goal_latent = self.gallery[0].to(DEVICE)
                        self.goal_set = True
                    else:
                        print(
                            "🎯 Mission Start: Encoding goal frame from Dataset [Ep 000]..."
                        )
                        self.agent.set_goal(episode_idx=0)
                        self.goal_set = True

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

                curr_model_state = self.map_sim_to_model_state(sim_state)

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

                info_dict = {"pixels": pixels_stacked, "action": actions_stacked}

                print(f"🧠 Step: Planning with 8k samples...")
                start_time = time.time()
                with torch.inference_mode():
                    planned_actions = self.solver.solve(info_dict)

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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--gallery", type=str, default=DEFAULT_GALLERY)
    args = parser.parse_args()

    server = LEWMInferenceServer(args.model, args.dataset, args.gallery)
    server.run()
