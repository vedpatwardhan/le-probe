import os
import datetime
import numpy as np
import zmq
import msgpack
import time
import argparse
import rerun as rr
import traceback
from PIL import Image
import mujoco
from simulation_base import GR1MuJoCoBase


class GR1VLAClient(GR1MuJoCoBase):
    """
    Proactive Autonomous Driver (REQ Client).
    Matches the Genesis simulation_gr1.py workflow.
    """

    def __init__(self, scene_path=None, server_host="localhost", server_port=5555):
        super().__init__(scene_path) if scene_path else super().__init__()

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
        """Captures 5-cam view (resized to 224x224) and calibrated state."""

        # ✅ Ensure scene and lighting are initialized before capture
        mujoco.mj_forward(self.model, self.data)

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        # ✅ POSTURAL CALIBRATION: Align Sim (-1.46) with Dataset (0.0)
        # We shift the Shoulder Pitch so 'Arms Down' looks like 'Neutral' to the model.
        state = self.get_state_32()
        state_calibrated = state.copy()
        state_calibrated[0] += 1.46  # Left Shoulder Pitch
        state_calibrated[16] += 1.46  # Right Shoulder Pitch

        obs = {"instruction": instruction, "state": pack_np(state_calibrated)}
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

        # ✅ VLA FIX: Initialize simulation state and renderer
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
                    valid_mask = ~np.isnan(actions_np)
                    if np.any(valid_mask):
                        self._debug_log(
                            f"🧠 Received Action Chunk: {actions_np.shape}. Stats -> [min:{np.nanmin(actions_np):.3f}, max:{np.nanmax(actions_np):.3f}, mean:{np.nanmean(actions_np):.3f}]"
                        )
                    else:
                        self._debug_log(
                            f"⚠️ Received Action Chunk: {actions_np.shape} BUT ALL VALUES ARE NAN!"
                        )

                    for i, action in enumerate(actions):
                        # Ensure 'action' is at least 1D (a vector of 32 joints)
                        action_np = np.array(action, dtype=np.float32)
                        if action_np.ndim == 0:
                            # If 'actions' was a 1D list, treat it as a single frame
                            action_32 = np.array(actions, dtype=np.float32)
                            action_32[0] -= 1.46
                            action_32[16] -= 1.46
                            self.dispatch_action(
                                action_32,
                                self.last_target_q,
                                n_steps=32,
                                reset_start=True,
                            )
                            break

                        action_32 = action_np
                        action_32[0] -= 1.46
                        action_32[16] -= 1.46

                        self.process_target_32(action_32)
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
