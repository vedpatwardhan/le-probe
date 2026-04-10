import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LEWMDataPlugin(torch.utils.data.Dataset):
    """
    A shim for LeRobotDataset to make it compatible with LeWorldModel's HDF5Dataset interface.
    Handles rescaling pixels from [0, 1] to [0, 255] and seq-slicing (T, C, H, W).
    """

    def __init__(
        self,
        repo_id,
        keys_to_load,
        num_steps=1,
        transform=None,
        use_virtual_actions=True,
    ):
        self.dataset = LeRobotDataset(repo_id, video_backend="pyav")
        self.keys_to_load = keys_to_load
        self.num_steps = num_steps
        self.transform = transform
        self.use_virtual_actions = use_virtual_actions

        # Mapping LeRobot keys to LeWM expected keys
        self.key_map = {
            "pixels": "observation.images.world_center",
            "action": "action",
            "state": "observation.state",
            "proprio": "observation.state",
        }

        # Optimization: Check if 'action' is already native to the dataset
        self.has_native_actions = "action" in self.dataset.hf_dataset.column_names
        if self.has_native_actions:
            print(
                "⚡ Detected pre-calculated actions in dataset. Using native action column."
            )

    def __len__(self):
        # We need an extra frame to compute the delta for the last action in a sequence if not native
        buffer = 1 if (self.use_virtual_actions and not self.has_native_actions) else 0
        return len(self.dataset) - (self.num_steps + buffer)

    def __getitem__(self, idx):
        """Returns a sequence of frames of length self.num_steps."""

        # 1. Handle Episode Boundaries: ensure T+1 steps are in the same episode
        # (Need T+1 frames to compute T deltas ONLY if not native)
        buffer = 1 if (self.use_virtual_actions and not self.has_native_actions) else 0
        curr_episode = self.dataset.hf_dataset[idx]["episode_index"]
        last_episode = self.dataset.hf_dataset[idx + self.num_steps + buffer - 1][
            "episode_index"
        ]

        if curr_episode != last_episode:
            # Shift the window back so the entire sequence is within the same episode
            idx = idx - self.num_steps
            if idx < 0:
                idx = 0

        # 2. Extract and slice keys
        batch = {}

        # We fetch T+1 states specifically if we need virtual actions ON THE FLY
        state_seq = []
        if (
            (self.use_virtual_actions and not self.has_native_actions)
            or "state" in self.keys_to_load
            or "proprio" in self.keys_to_load
        ):
            state_key = self.key_map["state"]
            # Fetch T+1 if we need to compute deltas
            fetch_len = (
                self.num_steps + 1
                if (self.use_virtual_actions and not self.has_native_actions)
                else self.num_steps
            )
            for i in range(idx, idx + fetch_len):
                state_seq.append(self.dataset[i][state_key])
            state_seq = torch.stack(state_seq)

        for target_key in self.keys_to_load:
            if (
                target_key == "action"
                and self.use_virtual_actions
                and not self.has_native_actions
            ):
                # COMPUTE VIRTUAL ACTIONS ON THE FLY: a_t = s_{t+1} - s_t
                # Slice to 32 to match GR-1 joint DoF and checkpoint size
                batch["action"] = (state_seq[1:] - state_seq[:-1])[..., :32]
                continue

            if (target_key == "state" or target_key == "proprio") and len(
                state_seq
            ) > 0:
                # Serve from pre-fetched sequence
                batch[target_key] = state_seq[: self.num_steps]
                continue

            # Standard sequence loading for other keys (pixels, or native action)
            source_key = self.key_map.get(target_key, target_key)
            seq = []
            for i in range(idx, idx + self.num_steps):
                try:
                    val = self.dataset[i][source_key]
                except Exception as e:
                    print(f"⚠️ Warning: Failed to decode frame {i}. Error: {e}")
                    val = self.dataset[idx][source_key]
                seq.append(val)

            batch[target_key] = torch.stack(seq)

        # 3. Apply LeWM transforms
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

        # If computing stats for virtual actions, we must use deltas ONLY IF NOT NATIVE
        if (
            col_name == "action"
            and self.use_virtual_actions
            and not self.has_native_actions
        ):
            state_key = self.key_map["state"]
            for i in range(0, sample_size - 1, max(1, sample_size // 1000)):
                s0 = self.dataset[i][state_key]
                s1 = self.dataset[i + 1][state_key]
                # Slice to 32 to match GR-1 joint DoF
                data_list.append((s1 - s0)[:32].numpy())
        else:
            # Standard path for scalar columns or native actions
            for i in range(0, sample_size, max(1, sample_size // 1000)):
                val = self.dataset[i][source_key]
                data_list.append(val.numpy())

        return np.stack(data_list)

    def get_dim(self, col_name):
        """Returns the dimensionality of the feature."""
        if col_name == "action" and self.use_virtual_actions:
            return 32

        source_key = self.key_map.get(col_name, col_name)
        # Pull a single frame to check shape
        sample = self.dataset[0][source_key]
        return sample.shape[-1]
