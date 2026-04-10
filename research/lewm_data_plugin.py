import torch
import numpy as np
import time
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
        self.repo_id = repo_id
        self.keys_to_load = keys_to_load
        self.num_steps = num_steps
        self.transform = transform
        self.use_virtual_actions = use_virtual_actions

        # Mapping LeRobot keys to LeWM expected keys
        self.key_map = {
            "observation.images.world_center": "observation.images.world_center",
            "observation.state": "observation.state",
            "action": "action",
        }
        self.key_map.update(
            {
                "pixels": "observation.images.world_center",
                "state": "observation.state",
                "proprio": "observation.state",
            }
        )

        # Base dataset initialization (no video backend yet to avoid fork issues)
        self.dataset = LeRobotDataset(repo_id)
        self._backend_initialized = False

        # Optimization: Check if 'action' is already native to the dataset
        self.has_native_actions = "action" in self.dataset.hf_dataset.column_names
        if self.has_native_actions:
            print(
                "⚡ Detected pre-calculated actions in dataset. Using native action column."
            )

        if self.keys_to_load is None:
            self.keys_to_load = list(self.key_map.keys())

    def _maybe_init_backend(self):
        """Worker-side initialization of the high-speed backend."""
        if self._backend_initialized:
            return

        try:
            import torchcodec

            self.dataset.video_backend = "torchcodec"
        except ImportError:
            self.dataset.video_backend = "pyav"

        self._backend_initialized = True

    @staticmethod
    def nest_dict(flat_dict):
        nested_dict = {}
        for k, v in flat_dict.items():
            parts = k.split(".")
            d = nested_dict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = v
        return nested_dict

    @staticmethod
    def flatten_dict(nested_dict, parent_key="", sep="."):
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(LEWMDataPlugin.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __len__(self):
        # We need an extra frame to compute the delta for the last action in a sequence if not native
        buffer = 1 if (self.use_virtual_actions and not self.has_native_actions) else 0
        return len(self.dataset) - (self.num_steps + buffer)

    def __getitem__(self, idx):
        """Returns a sequence of frames of length self.num_steps."""
        self._maybe_init_backend()

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
        # We fetch each index ONCE and pluck all required keys to avoid redundant video seeks.
        fetch_len = self.num_steps
        needs_plus_one_state = self.use_virtual_actions and not self.has_native_actions
        if needs_plus_one_state:
            fetch_len = self.num_steps + 1

        # Pre-fetch all samples in the sequence range
        samples = []
        for i in range(idx, idx + fetch_len):
            # Robust tiered retry for video decoding contention
            retry_attempts = 3
            sample_val = None
            for attempt in range(retry_attempts):
                try:
                    sample_val = self.dataset[i]
                    break
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        wait_time = 0.05 * (2**attempt)
                        time.sleep(wait_time)
                        continue
                    raise e
            samples.append(sample_val)

        # Distribute keys from pre-fetched samples
        batch = {}

        # Handle state specifically for virtual actions (T+1)
        if (
            needs_plus_one_state
            or "observation.state" in self.keys_to_load
            or "state" in self.keys_to_load
            or "proprio" in self.keys_to_load
        ):
            state_key = self.key_map["observation.state"]
            state_seq = torch.stack([s[state_key] for s in samples[:fetch_len]])
            # We'll handle slicing for virtual actions later in the existing logic if needed
            # but for now we put the full sequence in a temp place or handle batch directly.
            # Local logic below handles state_seq usage.

        # Load all requested keys
        for target_key in self.keys_to_load:
            source_key = self.key_map.get(target_key, target_key)
            # Take only the first num_steps if the key doesn't need the T+1 state
            seq_len = self.num_steps
            # Special case: virtual actions need T+1 states specifically
            batch[target_key] = torch.stack([s[source_key] for s in samples[:seq_len]])

        # 3. Apply LeWM transforms (expects nested structure)
        batch = {k: v for k, v in batch.items() if k in self.keys_to_load}
        nested_batch = self.nest_dict(batch)
        if self.transform:
            nested_batch = self.transform(nested_batch)

        # 4. Standardize for the Model: Flatten back and add Aliases
        # The transforms need nesting, but the model code expects flat keys like 'pixels'.
        final_batch = self.flatten_dict(nested_batch)

        # Ensure standard aliases exist for the forward pass in train_gr1.py
        # We collect aliases in a separate dict to avoid "dictionary changed size during iteration" error.
        aliases = {}
        for k, v in final_batch.items():
            if "world_center" in k:
                aliases["pixels"] = v
            if "observation.state" in k or (k == "state"):
                aliases["state"] = v
                aliases["proprio"] = v

        final_batch.update(aliases)

        return final_batch

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
                # Use full dimensionality from the dataset
                data_list.append((s1 - s0).numpy())
        else:
            # Standard path for scalar columns or native actions
            for i in range(0, sample_size, max(1, sample_size // 1000)):
                val = self.dataset[i][source_key]
                data_list.append(val.numpy())

        return np.stack(data_list)

    def get_dim(self, col_name):
        """Returns the dimensionality of the feature."""
        # Remove hardcoded 32-dim return. Let the dynamic check below handle it.

        source_key = self.key_map.get(col_name, col_name)
        # Pull a single frame to check shape
        sample = self.dataset[0][source_key]
        return sample.shape[-1]
