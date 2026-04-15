import os
import datetime
import numpy as np
import zmq
import msgpack
import time
import argparse
import rerun as rr
import json
import traceback
from PIL import Image
import mujoco
from simulation_base import GR1MuJoCoBase


class StatisticalUnscaler:
    """Grounds model outputs in dataset statistics instead of physical limits."""

    def __init__(self, stats_path):
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        # Extract Min/Max for Actions (32-dim)
        self.action_min = np.array(self.stats["action"]["min"], dtype=np.float32)
        self.action_max = np.array(self.stats["action"]["max"], dtype=np.float32)

        # Extract Min/Max for State Observations (32-dim)
        self.state_min = np.array(
            self.stats["observation.state"]["min"], dtype=np.float32
        )
        self.state_max = np.array(
            self.stats["observation.state"]["max"], dtype=np.float32
        )

    def unscale_action(self, norm_action):
        """Maps [0.0, 1.0] -> [stats_min, stats_max] Radians."""
        # Note: LeRobot Min-Max normalization uses (val - min) / (max - min)
        # So Un-normalization is: val * (max - min) + min
        return norm_action * (self.action_max - self.action_min) + self.action_min

    def scale_state(self, raw_state):
        """Maps Raw Radians -> [0.0, 1.0] using dataset min/max."""
        # The server's preprocessor does NOT apply Min-Max to the state — it expects
        # the client to pre-normalize. Evidence: Norm Range: [-1.46, 1.34] in server logs.
        denom = self.state_max - self.state_min
        # Clip denom to avoid division by zero for constant joints
        denom = np.where(denom < 1e-6, 1.0, denom)
        return np.clip((raw_state - self.state_min) / denom, 0.0, 1.0)


class GR1VLAClient(GR1MuJoCoBase):
    """
    Proactive Autonomous Driver (REQ Client).
    Matches the Genesis simulation_gr1.py workflow.
    """

    def __init__(self, scene_path=None, server_host="localhost", server_port=5555):
        super().__init__(scene_path) if scene_path else super().__init__()

        # ✅ Load Statistical Grounding
        stats_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "datasets/vedpatwardhan/gr1_pickup_compact_h264/meta/stats.json",
        )
        self.unscaler = StatisticalUnscaler(stats_path)

        # Enable VLA-specific diagnostic logging
        self.debug_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vla_debug.log"
        )
        with open(self.debug_log_path, "w") as f:
            f.write(f"--- VLA DEBUG LOG INITIALIZED: {datetime.datetime.now()} ---\n")

        self.vla_context = zmq.Context()
        self.vla_client = self.vla_context.socket(zmq.REQ)
        self.vla_client.setsockopt(zmq.RCVTIMEO, 120000)
        self.vla_client.connect(f"tcp://{server_host}:{server_port}")

    def capture_vla_observation(self, instruction):
        """Captures 5-cam view (resized to 224x224) and raw state."""
        mujoco.mj_forward(self.model, self.data)

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        # ✅ FULL STATISTICAL HANDSHAKE: Pre-normalize state before sending.
        # The server's preprocessor expects [0, 1] normalized inputs — it does NOT
        # re-scale the state. We apply the same Min-Max transform used during training.
        raw_state = self.get_state_32()
        norm_state = self.unscaler.scale_state(raw_state)

        obs = {"instruction": instruction, "state": pack_np(norm_state)}
        for cam in self.cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            img_raw = self.renderer.render()

            # ✅ RESIZE TO 224x224 (Model Native Resolution)
            img_resized = np.array(
                Image.fromarray(img_raw).resize((224, 224), Image.Resampling.LANCZOS)
            )
            obs[cam] = pack_np(img_resized)
        return obs

    def run(self, instruction="Pick up the red cube", max_chunks=10):
        print(f"🚀 Starting Autonomous Mission: '{instruction}'")
        self.reset_env()

        for chunk_idx in range(max_chunks):
            # 1. Perception
            obs_payload = self.capture_vla_observation(instruction)

            # 2. Planning
            print(
                f"[{time.strftime('%H:%M:%S')}] 🧠 Requesting Chunk {chunk_idx+1}/{max_chunks}..."
            )
            try:
                self.vla_client.send(msgpack.packb(obs_payload, use_bin_type=True))
                resp = msgpack.unpackb(self.vla_client.recv(), raw=False)

                if "action" in resp:
                    actions = resp["action"]
                    actions_np = np.array(actions, dtype=np.float32)

                    print(f"   🚀 Executing Chunk: {actions_np.shape} steps...")

                    for i, norm_action in enumerate(actions_np):
                        # ✅ STATISTICAL UNSCALING: Map [0, 1] back to Radians
                        # This replaces the manual action_32[0] -= 1.46 hacks.
                        action_32 = self.unscaler.unscale_action(norm_action)

                        self.process_target_32(action_32)

                        # Smooth Trajectory Threading
                        do_reset = i == 0
                        self.dispatch_action(
                            action_32,
                            self.last_target_q,
                            n_steps=32,
                            reset_start=do_reset,
                        )
                else:
                    print(f"❌ VLA Server Error: {resp.get('error', 'Unknown error')}")
                    break
            except Exception as e:
                print(f"❌ Mission Error: {e}")
                traceback.print_exc()
                break

        print("🏁 Mission Complete. Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR-1 Autonomous Mission Driver")
    parser.add_argument("--instruction", type=str, default="Pick up the red cube")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--host", type=str, default="localhost", help="VLA Server host")
    parser.add_argument("--port", type=int, default=5555, help="VLA Server port")
    args = parser.parse_args()

    # Re-init Rerun for standalone local run
    rr.init("gr1_vla", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    sim = GR1VLAClient(server_host=args.host, server_port=args.port)
    sim.run(instruction=args.instruction, max_chunks=args.chunks)
