import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure the library under test is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from research.lewm_data_plugin import LEWMDataPlugin


class MockLeRobotDataset:
    """Mock for lerobot.datasets.lerobot_dataset.LeRobotDataset"""

    def __init__(self, episodes):
        # episodes is a list of lengths (e.g. [10, 10])
        self.episodes = episodes
        self.indices = []
        for i, length in enumerate(episodes):
            for _ in range(length):
                self.indices.append({"episode_index": i})

        # State: simple linear sequence: state[i] = [i, i, i, ...]
        # This makes delta checking very easy (expected delta should be 1.0)
        self.num_states = sum(episodes)
        self.states = torch.stack(
            [torch.full((64,), float(i)) for i in range(self.num_states)]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {
            "observation.state": self.states[idx],
            "action": torch.zeros(64),
            "observation.images.world_center": torch.zeros(3, 224, 224),
        }

    @property
    def hf_dataset(self):
        return self.indices


@pytest.fixture
def mock_plugin():
    # Patch the LeRobotDataset class inside the plugin module
    with patch("research.lewm_data_plugin.LeRobotDataset") as mock_class:
        # Create a dataset with two episodes of 10 frames each
        mock_instance = MockLeRobotDataset(episodes=[10, 10])
        mock_class.return_value = mock_instance

        plugin = LEWMDataPlugin(
            repo_id="mock_repo",
            keys_to_load=["pixels", "action", "state"],
            num_steps=3,
            use_virtual_actions=True,
        )
        return plugin


def test_virtual_action_math(mock_plugin):
    """Verify that a_t = s_{t+1} - s_t."""
    # Request indices 0, 1, 2
    batch = mock_plugin[0]

    actions = batch["action"]
    states = batch["state"]

    # 1. Check shapes
    assert actions.shape == (3, 64)
    assert states.shape == (3, 64)

    # 2. Check delta math (s[idx] = idx, so delta should be 1.0)
    # Action[0] is state[1] - state[0]
    assert torch.all(actions[0] == 1.0)
    assert torch.all(actions[1] == 1.0)
    assert torch.all(actions[2] == 1.0)


def test_episode_boundary_handling(mock_plugin):
    """Verify that sequences never cross between episodes."""
    # Episode 0: [0 - 9], Episode 1: [10 - 19]
    # If we request idx=8, and num_steps=3:
    # We need frames idx to idx+num_steps (8, 9, 10, 11)
    # BUT 10 and 11 are in Episode 1.
    # Logic should shift the window back.

    idx = 8
    batch = mock_plugin[idx]

    # In the MockDataset, state[i] = i.
    # If the window shifted back appropriately (to 8 - 3 = 5),
    # then batch['state'] should contain states from [5, 6, 7].
    # Let's check the first state value.
    first_state_val = batch["state"][0, 0].item()

    assert first_state_val == (8 - 3)  # Should have shifted by num_steps

    # Ensure all actions in this batch are within the same episode (delta=1.0)
    assert torch.all(batch["action"] == 1.0)


def test_normalization_sampling_with_deltas(mock_plugin):
    """Verify that get_col_data correctly calculates deltas for stats."""
    # get_col_data('action') should return deltas, not the raw 'action' key
    data = mock_plugin.get_col_data("action")

    # Since our mock state increments by 1.0 every step,
    # the sample of actions should all be 1.0
    assert np.all(data == 1.0)
