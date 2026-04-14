import os
import datetime
import numpy as np
import zmq
import msgpack
import time
import argparse
import rerun as rr
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
        """Captures 5-cam view and state in the standard ZMQ 'Handshake' format."""

        # ✅ Ensure scene and lighting are initialized before capture
        mujoco.mj_forward(self.model, self.data)

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        state = self.get_state_32()
        obs = {"instruction": instruction, "state": pack_np(state)}
        for cam in self.cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            obs[cam] = pack_np(self.renderer.render())
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

                    print(f"   🚀 Executing {len(actions)} actions...")
                    for i, action in enumerate(actions):
                        action_32 = np.array(action, dtype=np.float32)
                        self.process_target_32(action_32)

                        # ✅ VLA FIX: Only reset start-point to actual physics on the FIRST action of the chunk.
                        # For frames 2-16, we follow the predicted trajectory smoothly.
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
                print(f"❌ Connection Error: {e}")
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
