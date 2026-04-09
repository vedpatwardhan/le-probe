import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# -----------------------------------------------------------------------------
# ROSETTA 64-DIM MAPPING (Simulation 32 -> Dataset 64)
# -----------------------------------------------------------------------------
# Compact 32: 0-6:L_Arm, 7-12:L_Hand, 13-15:Head, 16-22:R_Arm, 23-28:R_Hand, 29-31:Waist
ROSETTA_MAP = {
    # Arms
    **{i: i for i in range(7)},  # L-Arm: 0-6 -> 0-6
    **{i + 16: i + 7 for i in range(7)},  # R-Arm: 16-22 -> 7-13
    # Waist
    **{i + 29: i + 14 for i in range(3)},  # Waist: 29-31 -> 14-16
    # Head
    **{i + 13: i + 17 for i in range(3)},  # Head: 13-15 -> 17-19
    # Hands
    **{i + 7: i + 20 for i in range(6)},  # L-Hand: 7-12 -> 20-25
    **{i + 23: i + 32 for i in range(6)},  # R-Hand: 23-28 -> 32-37
}


class LeRobotManager:
    """Manages LeRobot dataset creation and frame buffering."""

    def __init__(
        self, repo_id="gr1_pickup_large", fps=10, root=None, upload_interval=20
    ):
        self.repo_id = repo_id
        self.fps = fps
        self.upload_interval = upload_interval
        self.episodes_since_sync = 0
        if root is None:
            # Default to 'datasets' folder at the cortex-gr1 root
            root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "datasets"
            )
        self.root = os.path.abspath(root)
        self.dataset = None
        self.episode_frame_count = 0

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
                "shape": (64,),
                "names": ["joints"],
            },
            "action": {"dtype": "float32", "shape": (64,), "names": ["joints"]},
        }

        # Point root directly to the specific dataset folder
        dataset_path = os.path.join(self.root, self.repo_id)
        metadata_path = os.path.join(dataset_path, "meta", "info.json")

        # We only resume if the folder exists AND has the metadata file.
        if not os.path.exists(metadata_path):
            # If the directory exists but has no metadata, we must use .create.
            # However, LeRobotDataset.create will fail if the folder exists with exist_ok=False.
            # We check if it's an empty (or partial) folder and handle it.
            if os.path.exists(dataset_path) and not os.listdir(dataset_path):
                print(
                    f"[LEROBOT] Warning: '{dataset_path}' exists but is not a valid dataset. Checking for cleanup..."
                )
                os.rmdir(dataset_path)

            os.makedirs(self.root, exist_ok=True)
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=dataset_path,
                features=features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4,
            )
        else:
            # Resume existing dataset using constructor
            self.dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=dataset_path,
            )
            # Standard constructor doesn't start writers for recording, do it manually
            if self.dataset.image_writer is None:
                self.dataset.start_image_writer(num_processes=0, num_threads=4)

        self.current_task = task_instruction
        print(
            f"[LEROBOT] Dataset '{self.repo_id}' ready. "
            f"Recording NEW episode for task: '{task_instruction}'"
        )

    def add_frame(self, views, state_32, action_32, reachability=None):
        """Remaps 32-dim simulation protocol to 64-dim Rosetta for the Hub."""
        if self.dataset is None:
            return
        self.episode_frame_count += 1

        # Create 64-dim buffers
        state_64 = np.zeros(64, dtype=np.float32)
        action_64 = np.zeros(64, dtype=np.float32)

        # Apply remapping
        for old_idx, new_idx in ROSETTA_MAP.items():
            state_64[new_idx] = state_32[old_idx]
            action_64[new_idx] = action_32[old_idx]

        # Grounding: Inject Reachability Volumes into unused state slots (62, 63)
        if reachability:
            state_64[62] = reachability.get("reachability/left_hand_volume", 0.0)
            state_64[63] = reachability.get("reachability/right_hand_volume", 0.0)

        # Add to LeRobot dataset features
        frame_data = {
            **{
                f"observation.images.{k}": Image.fromarray(v[..., :3].astype(np.uint8))
                for k, v in views.items()
            },
            "observation.state": state_64.astype(np.float32),
            "action": action_64.astype(np.float32),
            "task": self.current_task,
        }

        self.dataset.add_frame(frame_data)

    @property
    def total_episodes(self):
        """Returns the total number of episodes in the dataset."""
        return self._total_episodes

    @property
    def pending_uploads(self):
        """Returns the current number of background upload tasks in flight."""
        return self._pending_uploads

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

    def stop_episode(self):
        """Finalizes locally and and then conditionally queues a Hub sync."""
        if self.dataset is None:
            return

        print(f"[LEROBOT] Finalizing episode local save...")
        self.dataset.save_episode(parallel_encoding=False)
        self._total_episodes = self.dataset.num_episodes
        self.episodes_since_sync += 1
        print(f"[LEROBOT] Episode saved. Total frames: {self.episode_frame_count}")

        # Batch Sync: Periodic or and then conditional Hub synchronization
        self.episode_frame_count = 0  # Reset for next episode
        self.dataset = None

        if self.episodes_since_sync >= self.upload_interval:
            self.force_sync()
        else:
            print(
                f"[LEROBOT] Batch Status: {self.episodes_since_sync}/{self.upload_interval} episodes. Sync deferred."
            )

    def force_sync(self):
        """Manually triggers a motorized Hub synchronization of all new episodes."""
        if not LEROBOT_AVAILABLE:
            return

        # Use a temporary dataset object to trigger the push if not currently in an episode
        try:
            print(
                f"[LEROBOT] Initiating Hub synchronization (Batch size: {self.episodes_since_sync})..."
            )
            dataset_path = os.path.join(self.root, self.repo_id)
            temp_ds = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)

            self._pending_uploads += 1
            self.executor.submit(self._async_push_to_hub, temp_ds)
            self.episodes_since_sync = 0
        except Exception as e:
            print(f"[LEROBOT] ⚠️ Manual sync trigger failed: {e}")

    def discard_episode(self):
        """Aborts the current episode without saving or syncing."""
        if self.dataset is None:
            return
        print(
            f"[LEROBOT] Discarding current episode frames ({self.episode_frame_count})..."
        )
        self.episode_frame_count = 0  # Reset
        self.dataset = None  # Drop reference (temp data ignored)
