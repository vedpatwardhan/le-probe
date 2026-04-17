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
from gr1_protocol import StandardScaler


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
    ):
        super().__init__(scene_path) if scene_path else super().__init__()

        # ✅ Canonical Normalization (Min-Max [-1, 1])
        self.unscaler = StandardScaler()

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
        norm_state = np.clip(norm_state, -1.0, 1.0)

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
        print(f"🚀 Starting Autonomous Mission (Canonical Protocol): '{instruction}'")
        self.reset_env()

        # Audit History for Joint-Level verification
        audit_history = []

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

                    # ✅ ACTION SAFETY: Clip to [-1, 1] to match simulation_replay.py
                    actions_np = np.clip(actions, -1.0, 1.0)
                    print(
                        f"   🚀 Executing Chunk ({chunk_idx+1}/{max_chunks}): {actions_np.shape[0]} temporal steps..."
                    )

                    for i, norm_action in enumerate(actions_np):
                        # ✅ PROTOCOL ALIGNMENT: Pass normalized action directly to base.
                        # The base.process_target_32 handles the unscaling internally
                        # to ensure bit-perfect parity with simulation_replay.py.
                        self.process_target_32(norm_action)

                        # Record for joint-level audit
                        audit_history.append(
                            {
                                "chunk": chunk_idx,
                                "frame": i,
                                "brain_slots": norm_action.tolist(),
                                "sim_joints": self.get_state_32().tolist(),
                            }
                        )

                        # Smooth Trajectory Threading
                        do_reset = i == 0
                        self.dispatch_action(
                            norm_action,
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

        # Save Detailed Audit
        os.makedirs("inference_history_new", exist_ok=True)
        audit_path = "inference_history_new/joint_level_audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit_history, f)
        print(f"💾 Full joint-level audit saved to: {audit_path}")
        print("🏁 Mission Complete. Exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR-1 Autonomous Mission Driver")
    parser.add_argument("--instruction", type=str, default="Pick up the red cube")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--host", type=str, default="localhost", help="VLA Server host")
    parser.add_argument("--port", "-p", type=int, default=5555, help="VLA Server port")
    args = parser.parse_args()

    # Re-init Rerun for standalone local run
    rr.init("gr1_vla", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    sim = GR1VLAClient(server_host=args.host, server_port=args.port)
    sim.run(instruction=args.instruction, max_chunks=args.chunks)
