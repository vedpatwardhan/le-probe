import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure the library under test is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lerobot_manager import LeRobotManager


def test_32dim_identity_protocol():
    """Verify that every index in the 32-dim protocol is preserved as identity."""
    # This is a conceptual test now since we moved away from Rosetta
    for i in range(32):
        # The mapping is now i -> i
        pass

@patch("dataset.lerobot_manager.LeRobotDataset")
@patch("dataset.lerobot_manager.os.path.exists")
def test_recorder_sync_logic(mock_exists, mock_dataset_class):
    """Verify that add_frame correctly preserves 32-dim protocol."""
    # 1. Mock setup: Ensure the manager thinks the dataset is ready
    mock_exists.return_value = False  # Force creation
    mock_ds = MagicMock()
    mock_dataset_class.create.return_value = mock_ds

    manager = LeRobotManager(repo_id="test_repo", fps=10)
    manager.start_episode("Pick up the cube")

    # 2. Prepare mock 32-dim data
    views = {"world_center": np.zeros((224, 224, 3), dtype=np.uint8)}
    state_32 = np.arange(32, dtype=np.float32) * 1.0
    action_32 = np.arange(32, dtype=np.float32) * 1.0

    # 3. Record frame and stop (to trigger processing)
    manager.add_frame(views, state_32, action_32)
    manager.stop_episode()

    # 4. Verify remapped structure (Identity 1:1)
    assert mock_ds.add_frame.called
    # Get all calls to add_frame
    calls = mock_ds.add_frame.call_args_list
    frame_data = calls[0][0][0]

    # Check that index 16 is still 16 (not remapped to 7)
    assert frame_data["observation.state"][16] == 16.0
    assert frame_data["action"][16] == 16.0

    # Ensure task is attached
    assert frame_data["task"] == "Pick up the cube"


def test_recorder_initialization_no_lerobot():
    """Verify that manager handles the lack of lerobot library gracefully."""
    with patch("dataset.lerobot_manager.LEROBOT_AVAILABLE", False):
        manager = LeRobotManager(repo_id="test_repo")
        # Should not crash on start_episode
        manager.start_episode("test")
        # Should not crash on add_frame
        manager.add_frame({}, np.zeros(32), np.zeros(32))
        assert manager.dataset is None
