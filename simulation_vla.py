import numpy as np
import zmq
import msgpack
import time
import argparse
from simulation_base import GR1MuJoCoBase


class GR1VLAClient(GR1MuJoCoBase):
    """
    Proactive Autonomous Driver (REQ Client).
    Matches the Genesis simulation_gr1.py workflow.
    """

    def __init__(self, scene_path=None, server_port=5555):
        super().__init__(scene_path) if scene_path else super().__init__()
        self.vla_context = zmq.Context()
        self.vla_client = self.vla_context.socket(zmq.REQ)
        self.vla_client.setsockopt(zmq.RCVTIMEO, 30000)
        self.vla_client.connect(f"tcp://localhost:{server_port}")

    def capture_vla_observation(self, instruction):
        """Captures 5-cam view and state in the standard ZMQ 'Handshake' format."""

        def pack_np(arr):
            return {
                "data": arr.tobytes(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }

        obs = {"instruction": instruction, "state": pack_np(self.get_state_32())}
        for cam in self.cam_names:
            self.renderer.update_scene(self.data, camera=cam)
            obs[cam] = pack_np(self.renderer.render())
        return obs

    def run(self, instruction="Pick up the red cube", max_chunks=10):
        print(f"🚀 Starting Autonomous Mission: '{instruction}'")

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
                    print(f"   🚀 Executing {len(actions)} actions...")
                    for action in actions:
                        action_32 = np.array(action, dtype=np.float32)
                        self.process_target_32(action_32)
                        self.dispatch_action(action_32, self.last_target_q)
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
    args = parser.parse_args()

    # Re-init Rerun for standalone run
    import rerun as rr

    rr.init("gr1_vla", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    sim = GR1VLAClient()
    sim.run(instruction=args.instruction, max_chunks=args.chunks)
