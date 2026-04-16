import time
import numpy as np
import rerun as rr
import pandas as pd
import argparse
from simulation_base import GR1MuJoCoBase
from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS
from gr1_protocol import StandardScaler


class GR1ReplayClient(GR1MuJoCoBase):
    """
    Standardized Replay Client.
    Executes Parquet actions "As-Is" based on the canonical gr1_joint_order.txt.
    """

    def __init__(self, scene_path=None):
        super().__init__(scene_path or SCENE_PATH, restrict_ik=True)
        self.is_running = True

    def run(self, parquet_path):
        """
        High-Fidelity Replay of Dataset Shards.
        Assumes the parquet data contains 53 unique frames and follows the standardized joint order.
        """
        print(f"🎬 [Standard Replay] Loading {parquet_path}")

        try:
            df = pd.read_parquet(parquet_path)
            actions = df["action"].values
        except Exception as e:
            print(f"❌ Failed to load parquet: {e}")
            return

        # Reset environment
        self.reset_env()

        # Replay only the 4 specific keyframes requested by the user
        target_indices = [12, 25, 38, 51]
        print(f"📊 Replaying keyframes: {target_indices} (Normalized [-1, 1])...")

        for idx in target_indices:
            if idx >= len(actions):
                print(f"⚠️ Index {idx} out of range (max {len(actions)-1})")
                continue

            # Standardized 32-dim order (compact)
            action_32_norm = np.clip(actions[idx], -1.0, 1.0).astype(np.float32)

            print(f"🎬 Replaying Action Index {idx} (Normalized)...")

            # Calculate targets in Radians
            self.process_target_32(action_32_norm)

            # Execute using the UN-SCALED Radians action
            # This matches the teleop server's verified execution flow
            self.dispatch_action(
                action_32_norm,
                self.last_target_q,
                n_steps=200,
                render_freq=16,
                reset_start=False,
            )

            # Wait for user visualization
            time.sleep(2)

        print(f"🏁 Replay of {parquet_path} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR-1 Standardized Replay Tool")
    parser.add_argument(
        "--parquet", type=str, required=True, help="Path to Parquet shard for replay"
    )
    args = parser.parse_args()

    # Init Rerun
    rr.init("gr1_standard_replay", spawn=False)
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

    replay_sys = GR1ReplayClient()
    replay_sys.run(args.parquet)
