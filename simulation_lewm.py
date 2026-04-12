"""
ORACLE MPC SIMULATION DRIVER
Role: Client for the LEWM MPC Server. Drives the MuJoCo robot in closed-loop.
"""

import os
import datetime
import numpy as np
import zmq
import msgpack
import time
import argparse
import rerun as rr
from simulation_base import GR1MuJoCoBase


class GR1LEWMClient(GR1MuJoCoBase):
    def __init__(self, server_host="localhost", server_port=5555):
        super().__init__()

        # ZMQ Context
        self.context = zmq.Context()
        self.client = self.context.socket(zmq.REQ)
        self.client.setsockopt(zmq.RCVTIMEO, 120000)
        self.client.connect(f"tcp://{server_host}:{server_port}")

        print(f"🔗 Connected to MPC Server at {server_host}:{server_port}")

    def capture_observation(self, instruction):
        """Captures the canonical 'world_center' view and state."""

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        state = self.get_state_32()

        # We only need 'world_center' for Oracle perception
        self.renderer.update_scene(self.data, camera="world_center")
        img = self.renderer.render()

        return {
            "instruction": instruction,
            "state": pack_np(state),
            "world_center": pack_np(img),
        }

    def run(self, instruction="Pick up the red cube", max_steps=1000):
        print(f"🚀 Starting Omni-MPC Autonomous Mission: '{instruction}'")

        step_idx = 0
        while step_idx < max_steps:
            # 1. Perception
            obs_payload = self.capture_observation(instruction)

            # 2. Planning (Requesting the next optimized chunk)
            print(
                f"[{time.strftime('%H:%M:%S')}] 🧠 Requesting MPC Plan (Universal Gallery)..."
            )
            try:
                self.client.send(msgpack.packb(obs_payload, use_bin_type=True))
                resp = msgpack.unpackb(self.client.recv(), raw=False)

                if "action" in resp:
                    plan = np.array(resp["action"], dtype=np.float32)  # (Horizon, 32)
                    diag = resp.get("diagnostics", {})

                    print(
                        f"   🚀 Executing first action from plan (Solve Time: {diag.get('plan_time_ms')}ms)"
                    )

                    # For MPC, we usually execute only the FIRST action and re-plan
                    # Or we can execute a small chunk (e.g., 5 steps) to save time
                    curr_action = plan[0]
                    self.process_target_32(curr_action)
                    self.dispatch_action(curr_action, self.last_target_q)

                    step_idx += 1
                else:
                    print(f"❌ Server Error: {resp.get('error')}")
                    break
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    # Re-init Rerun for standalone local run (Matches simulation_vla.py pattern)
    rr.init("gr1_lewm", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    sim = GR1LEWMClient(server_host=args.host, server_port=args.port)
    sim.run()
