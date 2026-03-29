import os
import numpy as np
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


class LeRobotManager:
    """Manages LeRobot dataset creation and frame buffering."""

    def __init__(self, repo_id="gr1_pickup", fps=10, root="./datasets"):
        self.repo_id = repo_id
        self.fps = fps
        self.root = os.path.abspath(root)
        self.dataset = None
        self.episode_frame_count = 0

        if not LEROBOT_AVAILABLE:
            print("[WARNING] LeRobot not installed. Recording will be disabled.")
            return

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

    def add_frame(self, imgs_dict, state, action):
        """Adds a single frame to the current episode."""
        if self.dataset is None:
            return
        self.episode_frame_count += 1

        frame_data = {
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "task": self.current_task,
        }

        # Add images from the dict
        for key, img_array in imgs_dict.items():
            # Convert genesis [H, W, 4] or [H, W, 3] to PIL Image [H, W, 3]
            frame_data[f"observation.images.{key}"] = Image.fromarray(
                img_array[..., :3].astype(np.uint8)
            )

        self.dataset.add_frame(frame_data)

    def stop_episode(self):
        """Finalizes and saves the current episode."""
        if self.dataset is None:
            return

        self.dataset.save_episode(parallel_encoding=False)
        print(f"[LEROBOT] Episode saved. Total frames: {self.episode_frame_count}")
        # Finalize is only needed once technically, but save_episode handles the parquet write.
        print(f"[LEROBOT] Episode saved to {self.root}/{self.repo_id}")
        self.episode_frame_count = 0  # Reset for next episode
        self.dataset = None
