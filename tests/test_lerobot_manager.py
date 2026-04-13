import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from PIL import Image

# Absolute import from current location
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "research"))
from lerobot_manager import LeRobotManager, ROSETTA_MAP


@pytest.fixture
def mock_lerobot():
    """Mocks the LeRobot library components."""
    with patch("lerobot_manager.LEROBOT_AVAILABLE", True), patch(
        "lerobot_manager.LeRobotDataset"
    ) as mock_ds:

        # Mock the dataset instance
        instance = MagicMock()
        instance.frames = []
        instance.num_episodes = 0
        instance.image_writer = MagicMock()

        def mock_add_frame(data):
            instance.frames.append(data)

        instance.add_frame.side_effect = mock_add_frame
        instance.save_episode.side_effect = lambda **kwargs: setattr(
            instance, "num_episodes", instance.num_episodes + 1
        )

        mock_ds.create.return_value = instance
        mock_ds.return_value = instance

        yield instance


def test_action_smoothing(mock_lerobot):
    """Verifies that 'Staircase' actions are smoothed to 'Ramps' in the final dataset."""
    manager = LeRobotManager(repo_id="test_repo", root="/tmp/test_datasets")

    # 1. Start Episode
    manager.start_episode("Pick up the red cube")

    # 2. Add frames with a 'Staircase' action (constant target)
    # 10 frames of Joint 0 at target 0.5. State 0 starts at 0.0.
    for i in range(10):
        views = {"world_top": np.zeros((224, 224, 3))}
        state_32 = np.zeros(32)
        action_32 = np.zeros(32)
        action_32[0] = 0.5  # Constant target
        manager.add_frame(views, state_32, action_32)

    # 3. Stop Episode (triggers smoothing)
    manager.stop_episode()

    # 4. Assertions
    assert len(mock_lerobot.frames) == 10

    # Check the actions (Joint 0 is ROSETTA_MAP[0] = 0)
    actions = [f["action"][0] for f in mock_lerobot.frames]

    # Initial state was 0.0. Target was 0.5. Duration was 10.
    # Frame 0 should be 0.0 + (0.5 - 0.0) * (1/10) = 0.05
    # Frame 9 should be 0.5
    assert np.allclose(actions[0], 0.05, atol=1e-5)
    assert np.allclose(actions[9], 0.50, atol=1e-5)

    # Check for monotonicity (Smooth Ramp)
    for i in range(1, 10):
        assert (
            actions[i] > actions[i - 1]
        ), f"Action at frame {i} is not strictly increasing!"
        assert np.isclose(
            actions[i] - actions[i - 1], 0.05, atol=1e-5
        ), f"Step at frame {i} is not 0.05!"


def test_rosetta_remapping(mock_lerobot):
    """Verifies that the 32-dim simulation actions are correctly remapped to 64-dim Rosetta."""
    manager = LeRobotManager(repo_id="test_repo", root="/tmp/test_datasets")

    manager.start_episode("Remapping test")

    state_32 = np.zeros(32)
    action_32 = np.zeros(32)

    # Test specific mappings
    # Compact 29 (Waist Roll) -> Rosetta 14
    state_32[29] = 0.77
    action_32[29] = 0.88

    # Compact 16 (Right Shoulder Pitch) -> Rosetta 7
    state_32[16] = -0.11
    action_32[16] = -0.22

    manager.add_frame({"t": np.zeros((224, 224, 3))}, state_32, action_32)
    manager.stop_episode()

    final_frame = mock_lerobot.frames[0]
    final_state = final_frame["observation.state"]
    final_action = final_frame["action"]

    # Assert Waist (29 -> 14)
    assert final_state[14] == 0.77
    # Action 0.88 was smoothed against state 0.0.
    # With 1 frame, fraction = 1/1. action = 0 + (0.88 - 0) * 1 = 0.88.
    assert final_action[14] == 0.88

    # Assert Right Arm (16 -> 7)
    assert final_state[7] == -0.11
    assert final_action[7] == -0.22
