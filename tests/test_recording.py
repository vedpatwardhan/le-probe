import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure the library under test is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.lerobot_manager import LeRobotManager, ROSETTA_MAP


def test_rosetta_mapping_completeness():
    """Verify that every index in the 32-dim protocol has a unique mapping in ROSETTA."""
    dest_indices = set()
    for i in range(32):
        assert i in ROSETTA_MAP, f"Protocol Index {i} is missing a Rosetta mapping!"
        dest_idx = ROSETTA_MAP[i]
        assert (
            dest_idx not in dest_indices
        ), f"Colliding Rosetta mapping found for destination {dest_idx}"
        dest_indices.add(dest_idx)

    assert len(dest_indices) == 32


@patch("research.lerobot_manager.LeRobotDataset")
@patch("research.lerobot_manager.os.path.exists")
def test_recorder_sync_logic(mock_exists, mock_dataset_class):
    """Verify that add_frame correctly remaps simulation to hub protocol."""
    # 1. Mock setup: Ensure the manager thinks the dataset is ready
    mock_exists.return_value = False  # Force creation
    mock_ds = MagicMock()
    mock_dataset_class.create.return_value = mock_ds

    manager = LeRobotManager(repo_id="test_repo", fps=10)
    manager.start_episode("Pick up the cube")

    # 2. Prepare mock 32-dim data with unique values per index
    views = {"world_center": np.zeros((224, 224, 3), dtype=np.uint8)}
    state_32 = np.arange(32, dtype=np.float32) * 1.0
    action_32 = np.arange(32, dtype=np.float32) * 2.0

    # 3. Record frame
    manager.add_frame(views, state_32, action_32)

    # 4. Verify remapped structure
    assert mock_ds.add_frame.called
    frame_data = mock_ds.add_frame.call_args[0][0]

    # Check remapping for Left Arm (0 -> 0)
    assert frame_data["observation.state"][0] == 0.0
    assert frame_data["action"][0] == 0.0

    # Check remapping for Right Arm (16 -> 7)
    # In ROSETTA_MAP: 16 -> 7
    assert frame_data["observation.state"][7] == 16.0
    assert frame_data["action"][7] == 32.0  # 16.0 * 2.0

    # Ensure task is attached
    assert frame_data["task"] == "Pick up the cube"


def test_recorder_initialization_no_lerobot():
    """Verify that manager handles the lack of lerobot library gracefully."""
    with patch("research.lerobot_manager.LEROBOT_AVAILABLE", False):
        manager = LeRobotManager(repo_id="test_repo")
        # Should not crash on start_episode
        manager.start_episode("test")
        # Should not crash on add_frame
        manager.add_frame({}, np.zeros(32), np.zeros(32))
        assert manager.dataset is None
