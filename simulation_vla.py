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


class StandardScaler:
    """Grounds model outputs in either dataset statistics (Z-Score) or physical limits (Min-Max)."""

    def __init__(self, stats_path=None, mode="compact"):
        self.mode = mode
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
            # Z-Score statistics
            self.state_mean = np.array(
                self.stats["observation.state"]["mean"], dtype=np.float32
            )
            self.state_std = np.array(
                self.stats["observation.state"]["std"], dtype=np.float32
            )
            # Min-Max statistics (fallback)
            self.action_min = np.array(self.stats["action"]["min"], dtype=np.float32)
            self.action_max = np.array(self.stats["action"]["max"], dtype=np.float32)
        else:
            print(
                "[WARNING] No stats.json found. Defaulting to Physical Limits (Base)."
            )
            self.mode = "base"

    def unscale_action(self, norm_action):
        """Maps model output to Radians."""
        if self.mode == "compact":
            # Per fine-tune config: "ACTION": "IDENTITY" (model predicts raw radians)
            return np.array(norm_action, dtype=np.float32)
        else:
            # Per base model protocol: Maps [-1, 1] to JOINT_LIMITS
            # Note: We use JOINT_LIMITS from gr1_config
            from gr1_config import JOINT_LIMITS_MIN, JOINT_LIMITS_MAX

            lmin = np.array(JOINT_LIMITS_MIN, dtype=np.float32)
            lmax = np.array(JOINT_LIMITS_MAX, dtype=np.float32)
            # Simple Min-Max unscaling: val * (max - min) / 2 + (max + min) / 2
            # because the model output is in [-1, 1]
            return (norm_action + 1.0) * (lmax - lmin) / 2.0 + lmin

    def scale_state(self, raw_state):
        """Maps Raw Radians to model-ready distribution."""
        if self.mode == "compact":
            # Per fine-tune config: "STATE": "MEAN_STD" (Z-Score)
            safe_std = np.where(self.state_std > 1e-6, self.state_std, 1.0)
            return (raw_state - self.state_mean) / safe_std
        else:
            # Per base model protocol: Maps Radians to [-1, 1]
            from gr1_config import JOINT_LIMITS_MIN, JOINT_LIMITS_MAX

            lmin = np.array(JOINT_LIMITS_MIN, dtype=np.float32)
            lmax = np.array(JOINT_LIMITS_MAX, dtype=np.float32)
            range_val = lmax - lmin
            range_val = np.where(range_val < 1e-6, 1.0, range_val)
            # Maps [min, max] to [-1, 1]
            return 2.0 * (raw_state - lmin) / range_val - 1.0


class GR1VLAClient(GR1MuJoCoBase):
    """
    Proactive Autonomous Driver (REQ Client).
    Matches the Genesis simulation_gr1.py workflow.
    """

    def __init__(
        self,
        scene_path=None,
        server_host="localhost",
        server_port=5555,
        mode="compact",
    ):
        super().__init__(scene_path) if scene_path else super().__init__()
        self.mode = mode

        # ✅ Load Statistical Grounding
        stats_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "datasets/vedpatwardhan/gr1_pickup_compact_h264/meta/stats.json",
        )
        self.unscaler = StandardScaler(stats_path, mode=self.mode)

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

        # ✅ UNIVERSAL HANDSHAKE: Scale state based on mode
        raw_state = self.get_state_32()
        norm_state = self.unscaler.scale_state(raw_state)

        obs = {"instruction": instruction, "state": pack_np(norm_state)}
        for cam in self.cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            img_raw = self.renderer.render()

            # ✅ RESIZE TO 224x224 (Model Native Resolution)
            # Using Image.Resampling.LANCZOS to match LeRobot data augmentation pipelines
            img_resized = np.array(
                Image.fromarray(img_raw).resize((224, 224), Image.Resampling.LANCZOS)
            )
            obs[cam] = pack_np(img_resized)
        return obs

    def run(self, instruction="Pick up the red cube", max_chunks=10):
        print(f"🚀 Starting Autonomous Mission ({self.mode} protocol): '{instruction}'")
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
                        # ✅ UNSCALE ACTION: Map model output back to Radians
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "compact"],
        default="compact",
        help="Protocol mode: 'base' for raw limits, 'compact' for Z-Score stats",
    )
    args = parser.parse_args()

    # Re-init Rerun for standalone local run
    rr.init("gr1_vla", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    sim = GR1VLAClient(server_host=args.host, server_port=args.port, mode=args.mode)
    sim.run(instruction=args.instruction, max_chunks=args.chunks)
