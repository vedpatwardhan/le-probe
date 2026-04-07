import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LEWMDataPlugin(torch.utils.data.Dataset):
    """
    A shim for LeRobotDataset to make it compatible with LeWorldModel's HDF5Dataset interface.
    Handles rescaling pixels from [0, 1] to [0, 255] on-the-fly.
    """

    def __init__(self, repo_id, keys_to_load, transform=None):
        self.dataset = LeRobotDataset(repo_id)
        self.keys_to_load = keys_to_load
        self.transform = transform

        # Mapping LeRobot keys to LeWM expected keys
        self.key_map = {
            "pixels": "observation.images.world_center",
            "action": "action",
            "state": "observation.state",
            "proprio": "observation.state",
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # LeRobot [idx] returns a dict of tensors
        item = self.dataset[idx]

        # Extract and rename keys
        batch = {}
        for target_key in self.keys_to_load:
            source_key = self.key_map.get(target_key, target_key)
            if source_key in item:
                val = item[source_key]

                # SPECIAL RANGE CONVERSION: Rescale pixels 0-1 -> 0-255
                if target_key == "pixels":
                    val = val * 255.0

                batch[target_key] = val

        # Apply LeWM transforms (ImageNet normalizer, etc.)
        if self.transform:
            batch = self.transform(batch)

        return batch

    def get_col_data(self, col_name):
        """Used by LeWM's get_column_normalizer to compute offline stats."""
        source_key = self.key_map.get(col_name, col_name)
        print(f"📊 Scanning dataset for column: {col_name} (source: {source_key})")

        # LeRobot stores metrics in its own parquet files, but for a 150-episode dataset,
        # we can just pull a large sample to compute stable stats.
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
