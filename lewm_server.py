"""
ORACLE MPC INFERENCE SERVER
Role: Standalone ZMQ server hosting the JEPA world model and CEM solver.

Key Features:
1. Internal History Buffering (maintains the 3-frame sliding window).
2. Rosetta Mapping (32-dim simulator joints -> 64-dim dataset manifold).
3. Calibrated Search (8k samples, 3.0 variance, 10x cost pressure).
"""

import zmq
import msgpack
import torch
import numpy as np
import time
import argparse
from pathlib import Path

# Local imports from the research layer
from research.goal_mapper import GoalMapper
from stable_worldmodel.solver.cem import CEMSolver

# Configuration Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 5555


class LEWMInferenceServer:
    def __init__(self, model_path, dataset_root):
        print("--- Initializing Oracle MPC Server ---")

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
        # Use our 15-step horizon that passed the diagnostic
        self.solver.configure(horizon=15)

        # 3. State/History Buffering (Wait for Mission Start)
        self.history = {
            "pixels": [],  # List of (3, 224, 224) tensors
            "actions": [],  # List of (64,) action vectors
        }
        self.goal_set = False

    def map_sim_to_model_state(self, state_32):
        """
        Rosetta Mapping: Translates 32-dim MuJoCo joints to 64-dim Dataset manifold.
        Aligned with gr00t_server.py logic.
        """
        state_full = np.zeros(64, dtype=np.float32)
        state_full[0:7] = state_32[0:7]  # Left Arm
        state_full[7:13] = state_32[7:13]  # Left Hand
        state_full[19:22] = state_32[13:16]  # Neck/Head
        state_full[22:29] = state_32[16:23]  # Right Arm
        state_full[29:35] = state_32[23:29]  # Right Hand
        state_full[41:44] = state_32[29:32]  # Waist
        return state_full

    def run(self, host="0.0.0.0"):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{PORT}")
        print(f"🚀 LEWM MPC Server listening on port {PORT}...")

        while True:
            try:
                message = socket.recv()
                req = msgpack.unpackb(message, raw=False)

                # 1. Reset/Goal Logic
                if not self.goal_set:
                    print("🎯 Setting Initial Goal (Proxy: Episode 000)...")
                    # In a real run, we might use a dynamic image from the client
                    self.agent.set_goal(episode_idx=0)
                    self.goal_set = True

                # 2. Unpack Raw Inputs
                def unpack_np(d):
                    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(
                        d["shape"]
                    )

                # We use the world_center view (canonical for Oracle training)
                raw_image = unpack_np(req.get("world_center"))
                sim_state = unpack_np(req.get("state"))

                # 3. Preprocess Image: (H, W, 3) -> (3, 224, 224)
                # Map to official training transforms
                batch = self.agent.transform({"pixels": raw_image})
                image_t = batch["pixels"].to(DEVICE)

                # 4. Update History Buffer
                self.history["pixels"].append(image_t)
                if len(self.history["pixels"]) > 3:
                    self.history["pixels"].pop(0)

                # Current state mapping
                curr_model_state = self.map_sim_to_model_state(sim_state)

                # 5. Handle Action History
                # If we don't have enough history, pad with current state or zeros
                while len(self.history["actions"]) < 3:
                    self.history["actions"].append(np.zeros(64, dtype=np.float32))

                # 6. Planning (Imagination)
                # CEM needs: [pixels: (1, 1, 3, 3, 224, 224), action: (1, 1, 3, 64)]
                pixels_stacked = (
                    torch.stack(self.history["pixels"]).unsqueeze(0).unsqueeze(0)
                )  # (1, 1, 3, 3, 224, 224)
                actions_stacked = (
                    torch.tensor(np.stack(self.history["actions"]), dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                info_dict = {"pixels": pixels_stacked, "action": actions_stacked}

                print(f"🧠 Planning with {pixels_stacked.shape} stack...")
                start_time = time.time()
                with torch.inference_mode():
                    # Solve for the best action chunk
                    # CEM returns (n_steps, horizon, adim)
                    planned_actions = self.solver.solve(info_dict)

                # CEM returns (1, 15, 64) for n_steps=1
                best_plan = planned_actions[0].cpu().numpy()  # (15, 64)

                plan_time = time.time() - start_time
                print(f"   [DEBUG] Plan Search Time: {plan_time:.2f}s")

                # 7. Update Action History (with the FIRST action of the plan)
                # This ensures the next step has the transition context
                self.history["actions"].append(best_plan[0])
                if len(self.history["actions"]) > 3:
                    self.history["actions"].pop(0)

                # 8. Return Actions to Simulator
                # Simulator expects 32-dim joints. We map back or let it slice.
                # Here we just return the full plan, simulator will slice top 32.
                payload = {
                    "action": best_plan.tolist(),
                    "diagnostics": {
                        "plan_time_ms": int(plan_time * 1000),
                        "horizon": best_plan.shape[0],
                    },
                }
                socket.send(msgpack.packb(payload, use_bin_type=True))

            except Exception as e:
                print(f"❌ Server Error: {e}")
                import traceback

                traceback.print_exc()
                socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/.cache/huggingface/hub/datasets--vedpatwardhan--gr1_pickup_processed/snapshots/55d7c8005e72628b67219082db433a70bfbf3c28",
    )
    args = parser.parse_args()

    server = LEWMInferenceServer(args.model, args.dataset)
    server.run()
