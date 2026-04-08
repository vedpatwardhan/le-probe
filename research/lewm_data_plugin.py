import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LEWMDataPlugin(torch.utils.data.Dataset):
    """
    A shim for LeRobotDataset to make it compatible with LeWorldModel's HDF5Dataset interface.
    Handles rescaling pixels from [0, 1] to [0, 255] and seq-slicing (T, C, H, W).
    """

    def __init__(self, repo_id, keys_to_load, num_steps=1, transform=None):
        self.dataset = LeRobotDataset(repo_id)
        self.keys_to_load = keys_to_load
        self.num_steps = num_steps
        self.transform = transform

        # Mapping LeRobot keys to LeWM expected keys
        self.key_map = {
            "pixels": "observation.images.world_center",
            "action": "action",
            "state": "observation.state",
            "proprio": "observation.state",
        }

    def __len__(self):
        # We must ensure there's enough room for a full sequence
        return len(self.dataset) - self.num_steps

    def __getitem__(self, idx):
        """Returns a sequence of frames of length self.num_steps."""

        # 1. Handle Episode Boundaries: ensure all steps are in the same episode
        # LeRobotDataset stores indices in its 'hf_dataset'
        curr_episode = self.dataset.hf_dataset[idx]["episode_index"]
        last_episode = self.dataset.hf_dataset[idx + self.num_steps - 1][
            "episode_index"
        ]

        if curr_episode != last_episode:
            # If we cross an episode, shift the index back so we stay within the current episode
            # This is a standard strategy for world model temporal slicing
            idx = idx - (self.num_steps - 1)
            if idx < 0:
                idx = 0

        # 2. Extract and slice keys
        batch = {}
        for target_key in self.keys_to_load:
            source_key = self.key_map.get(target_key, target_key)

            # Pull the sequence [idx : idx + num_steps]
            seq = []
            for i in range(idx, idx + self.num_steps):
                val = self.dataset[i][source_key]

                # SPECIAL RANGE CONVERSION: Rescale pixels 0-1 -> 0-255
                if target_key == "pixels":
                    val = val * 255.0

                seq.append(val)

            # Stack into (T, ...)
            batch[target_key] = torch.stack(seq)

        # 3. Apply LeWM transforms (ImageNet normalizer, etc.)
        # Transformations in LeWM expect dicts of (T, ...) or (B, T, ...)
        if self.transform:
            batch = self.transform(batch)

        return batch

    def get_col_data(self, col_name):
        """Used by LeWM's get_column_normalizer to compute offline stats."""
        source_key = self.key_map.get(col_name, col_name)
        print(f"📊 Scanning dataset for column: {col_name} (source: {source_key})")

        # Pull a large sample to compute stable stats.
        sample_size = min(5000, len(self.dataset))
        data_list = []
        for i in range(0, sample_size, max(1, sample_size // 1000)):
            val = self.dataset[i][source_key]
            data_list.append(val.numpy())

        return np.stack(data_list)

    def get_dim(self, col_name):
        """Returns the dimensionality of the feature."""
        source_key = self.key_map.get(col_name, col_name)
        # Pull a single frame to check shape
        sample = self.dataset[0][source_key]
        return sample.shape[-1]
