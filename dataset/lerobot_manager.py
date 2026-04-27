
# --- Path Stabilization ---
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import shutil
import json
import lerobot.datasets.utils as lerobot_utils
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from gr1_protocol import StandardScaler

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# -----------------------------------------------------------------------------
# 32-DIM PROTOCOL ENFORCEMENT
# -----------------------------------------------------------------------------
# Both state and action are recorded as raw 32-dim vectors mirroring gr1_joint_order.txt.


class LeRobotManager:
    """Manages LeRobot dataset creation and frame buffering."""

    def __init__(
        self, repo_id="gr1_pickup_grasp", fps=10, root=None, upload_interval=20
    ):
        self.repo_id = repo_id
        self.fps = fps
        self.upload_interval = upload_interval
        self.episodes_since_sync = 0
        if root is None:
            # Default to 'datasets' folder at the le-probe root
            root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "datasets"
            )
        self.root = os.path.abspath(root)
        self.dataset = None
        self.episode_frame_count = 0
        self.episode_buffer = []  # Buffer for post-recording smoothing

        if not LEROBOT_AVAILABLE:
            print("[WARNING] LeRobot not installed. Recording will be disabled.")
            return

        # Sequential Uploader: Max workers = 1 to ensure repo-level synchronization parity
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._pending_uploads = 0
        self._total_episodes = 0

        # Scan for existing episodes to initialize counter
        try:
            chunk_path = os.path.join(self.root, self.repo_id, "data", "chunk-000")
            if os.path.exists(chunk_path):
                existing = [f for f in os.listdir(chunk_path) if f.endswith(".parquet")]
                self._total_episodes = len(existing)
        except Exception as e:
            print(f"[LEROBOT] Warning: Could not scan existing episodes: {e}")

        # Initial probe of dataset scale
        dataset_path = os.path.join(self.root, self.repo_id)
        if LEROBOT_AVAILABLE and os.path.exists(
            os.path.join(dataset_path, "meta", "info.json")
        ):
            try:
                temp_ds = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
                self._total_episodes = temp_ds.num_episodes
            except Exception as e:
                print(f"[LEROBOT] ⚠️ Metadata probe failed: {e}")

        # Canonical Scaling Logic
        self.unscaler = StandardScaler()

    def start_episode(self, task_instruction):
        """Initializes a new episode or resumes the dataset."""
        if not LEROBOT_AVAILABLE:
            return

        features = {
            "observation.images.world_top": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.world_left": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.world_right": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.world_center": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.world_wrist": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (32,),
                "names": ["joints"],
            },
            "action": {"dtype": "float32", "shape": (32,), "names": ["joints"]},
            "observation.physics": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["cube_z", "is_grasping", "target_dist"],
            },
        }

        # Point root directly to the specific dataset folder
        dataset_path = os.path.join(self.root, self.repo_id)
        metadata_path = os.path.join(dataset_path, "meta", "info.json")
        episodes_path = os.path.join(dataset_path, "meta", "episodes")

        # We only resume if the folder exists AND is a fully initialized dataset.
        # If the folders are present but empty/incomplete, we MUST use .create.
        if not os.path.exists(metadata_path) or not os.path.exists(episodes_path):
            # If the directory exists but is incomplete, we clean up for a fresh creation.

            os.makedirs(self.root, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=dataset_path,
                features=features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4,
                video_backend="ffmpeg",
                vcodec="h264",
            )
        else:
            # Resume existing local dataset
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=dataset_path,
                vcodec="h264",
            )
            # Standard constructor doesn't start writers for recording, do it manually
            if self.dataset.image_writer is None:
                self.dataset.start_image_writer(num_processes=0, num_threads=4)

        self.current_task = task_instruction
        self.episode_buffer = []  # Clear buffer for new episode
        print(
            f"[LEROBOT] Dataset '{self.repo_id}' ready. "
            f"Recording NEW episode for task: '{task_instruction}'"
        )

    def add_frame(self, views, state_32, action_32, extra_info=None):
        """Buffers a frame for post-episode processing (to prevent simulation lag)."""
        if self.dataset is None:
            return
        self.episode_frame_count += 1
        self.episode_buffer.append(
            {
                "views": views,
                "state_32": state_32.copy() if hasattr(state_32, "copy") else state_32,
                "action_32": (
                    action_32.copy() if hasattr(action_32, "copy") else action_32
                ),
                "extra_info": extra_info
                or {
                    "cube_z": 0.0,
                    "is_grasping": False,
                    "target_dist": 1.0,
                },
            }
        )

    def stop_episode(self):
        """Finalizes the episode locally by applying smoothing and flushing to the dataset."""
        if self.dataset is None or not self.episode_buffer:
            print("[LEROBOT] Alert: stop_episode called on empty buffer.")
            return

        print(
            f"[LEROBOT] Applying Smooth Absolute Interpolation to {len(self.episode_buffer)} frames..."
        )

        # 1. Smooth Absolute Interpolation (Linear Ramp Integration)
        # First reference is the proprioception of the first frame
        prev_target = self.episode_buffer[0]["state_32"].copy()
        raw_targets = [f["action_32"] for f in self.episode_buffer]
        smoothed_actions_32 = [None] * len(self.episode_buffer)

        i = 0
        while i < len(raw_targets):
            curr_target = raw_targets[i]

            # Find end of constant-target block (Staircase segment)
            j = i
            while j < len(raw_targets) and np.allclose(
                raw_targets[j], curr_target, atol=1e-6
            ):
                j += 1

            duration = j - i
            start_pos = prev_target
            end_pos = curr_target

            for k in range(duration):
                fraction = (k + 1) / duration
                interpolated_pos = start_pos + (end_pos - start_pos) * fraction
                smoothed_actions_32[i + k] = interpolated_pos.astype(np.float32)

                # [AUDIT:STAGE3] Data Buffer for R-Shoulder Roll
                print(f"[AUDIT:STAGE3] {interpolated_pos[17]:.6f}")

            prev_target = curr_target.copy()
            i = j

        # 2. Canonical Normalize and Remap to 64-dim Rosetta and Flush to LeRobotDataset
        self.unscaler = StandardScaler()

        print(f"[LEROBOT] Finalizing episode local save with 32-dim Normalization...")
        for idx, frame in enumerate(self.episode_buffer):
            # Normalize raw radians to [-1, 1] based on canonical physical limits
            norm_state_32 = self.unscaler.scale_state(frame["state_32"])
            norm_action_32 = self.unscaler.scale_state(smoothed_actions_32[idx])

            # Remap in-memory frames to LeRobot features
            frame_data = {
                **{
                    f"observation.images.{k}": Image.fromarray(
                        v[..., :3].astype(np.uint8)
                    )
                    for k, v in frame["views"].items()
                },
                "observation.state": norm_state_32.astype(np.float32),
                "action": norm_action_32.astype(np.float32),
                "observation.physics": np.array(
                    [
                        float(frame["extra_info"].get("cube_z", 0.0)),
                        float(frame["extra_info"].get("is_grasping", False)),
                        float(frame["extra_info"].get("target_dist", 1.0)),
                    ],
                    dtype=np.float32,
                ),
                "task": self.current_task,
            }

            self.dataset.add_frame(frame_data)

        # Commit to Parquet (Main Dataset)
        self.dataset.save_episode(parallel_encoding=False)

        # Incremental Reward calculation for the episode
        self._append_rewards_to_sidecar()
        self._total_episodes = self.dataset.num_episodes
        self.episodes_since_sync += 1
        print(f"[LEROBOT] Episode saved. Total frames: {len(self.episode_buffer)}")

        # Reset buffers and state for next episode
        self.episode_buffer = []
        self.episode_frame_count = 0
        self.dataset = None

        # Check for batch sync trigger
        if self.episodes_since_sync >= self.upload_interval:
            self.force_sync()

    def discard_episode(self):
        """Aborts the current episode without saving or syncing."""
        if self.dataset is None:
            return
        print(
            f"[LEROBOT] Discarding current episode buffer ({len(self.episode_buffer)} frames)..."
        )
        self.episode_buffer = []
        self.episode_frame_count = 0
        self.dataset = None

    def force_sync(self):
        """Manually triggers a motorized Hub synchronization of all new episodes."""
        if not LEROBOT_AVAILABLE:
            return
        try:
            print(f"[LEROBOT] Initiating Hub synchronization...")
            dataset_path = os.path.join(self.root, self.repo_id)
            temp_ds = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)
            self._pending_uploads += 1
            self.executor.submit(self._async_push_to_hub, temp_ds)
            self.episodes_since_sync = 0
        except Exception as e:
            print(f"[LEROBOT] ⚠️ Manual sync trigger failed: {e}")

    def _async_push_to_hub(self, dataset_to_push):
        """Internal worker to execute the hub push operation."""
        try:
            print(f"[SYNC] Uploading current batch to Hub: {self.repo_id}...")
            dataset_to_push.push_to_hub()
            print(f"[SYNC] Hub synchronization successful.")
        except Exception as e:
            print(f"[SYNC] ⚠️ Hub Upload Failed: {e}")
        finally:
            self._pending_uploads = max(0, self._pending_uploads - 1)

    def _append_rewards_to_sidecar(self):
        """Calculates rewards for the current buffer and appends to progress_sparse.parquet."""
        if not LEROBOT_AVAILABLE:
            return

        print(
            f"[LEROBOT] Appending rewards to sidecar for episode {self.dataset.num_episodes - 1}..."
        )

        # Determine global start index for this episode
        # num_frames includes the frames we JUST added
        num_total_frames = self.dataset.num_frames
        episode_len = len(self.episode_buffer)
        start_idx = num_total_frames - episode_len
        episode_idx = self.dataset.num_episodes - 1

        rewards = []
        MAX_APPROACH_DIST = 0.5

        for i, frame in enumerate(self.episode_buffer):
            # Extract physics from extra_info
            cube_z = float(frame["extra_info"].get("cube_z", 0.0))
            is_grasping = bool(frame["extra_info"].get("is_grasping", False))
            target_dist = float(frame["extra_info"].get("target_dist", 1.0))

            # -----------------------------------------------------------------
            # 32-FRAME PURE INVERTED DISTANCE STRATEGY
            # -----------------------------------------------------------------
            if len(self.episode_buffer) == 32:
                # The entire reward is just the proximity to the target (scaled to 10.0)
                # target_dist is usually around 0.5 to 1.0 at start, and ~0.0 at grasp.
                proximity = max(0, 1.0 - target_dist)
                score = proximity * 10.0

            else:
                # Fallback Heuristic for variable-length/manual teleop
                z_offset = cube_z - 0.82
                if z_offset > 0.03:
                    score = 10.0
                elif is_grasping:
                    score = 0.6
                else:
                    dist_factor = max(0, 1.0 - (target_dist / MAX_APPROACH_DIST))
                    score = 0.1 + (dist_factor * 0.4)

            # Final Temporal Slope for unique indexing
            final_score = float(score + (i * 0.0001))

            rewards.append(
                {
                    "index": start_idx + i,
                    "episode_index": episode_idx,
                    "progress_sparse": final_score,
                    "progress_dense": final_score,
                }
            )

        reward_df = pd.DataFrame(rewards)
        sidecar_path = os.path.join(self.root, self.repo_id, "progress_sparse.parquet")

        if os.path.exists(sidecar_path):
            existing_df = pd.read_parquet(sidecar_path)
            # Ensure we don't have overlapping indices if a save failed previously
            existing_df = existing_df[existing_df["index"] < start_idx]
            reward_df = pd.concat([existing_df, reward_df], ignore_index=True)

        reward_df.to_parquet(sidecar_path)
        print(
            f"[LEROBOT] Sidecar updated with Smart Rewards. Total reward frames: {len(reward_df)}"
        )

    @property
    def total_episodes(self):
        return self._total_episodes

    @property
    def pending_uploads(self):
        return self._pending_uploads