import torch
import torch.nn as nn
import pytest
import os
import shutil
import unittest.mock as mock
from lewm.metrics import MetricsCallback


# Mock classes to simulate PyTorch Lightning structure
class MockModule(nn.Module):
    def __init__(self):
        super().__init__()

    def log(self, name, value, **kwargs):
        pass


class MockTrainer:
    def __init__(self):
        self.global_step = 10
        self.current_epoch = 1


@mock.patch("wandb.log")
@mock.patch("wandb.Image")
def test_csv_logging(mock_wandb_image, mock_wandb_log):
    """
    Verifies that MetricsCallback correctly initializes and writes
    manifold diagnostics to a persistent CSV file.
    """
    csv_file = "test_diagnostics.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)

    # Initialize callback
    callback = MetricsCallback(log_every_n_steps=1, csv_path=csv_file)
    trainer = MockTrainer()
    pl_module = MockModule()

    # Mock data (B, T, D)
    B, T, D = 4, 3, 192
    emb = torch.randn(B, T, D)
    batch = {"action": torch.randn(B, T, 32) * 0.1}
    outputs = {"emb": emb, "loss": torch.tensor(0.5)}

    # Trigger callback
    callback.on_train_batch_end(trainer, pl_module, outputs, batch, 0)

    # Verify CSV existence and content
    assert os.path.exists(csv_file), "CSV file was not created"

    with open(csv_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2, "CSV should have header and one data row"
        header = lines[0].strip().split(",")
        data = lines[1].strip().split(",")

        # Check critical headers
        assert "soft_rank" in header
        assert "action_mag" in header
        assert "s_0" in header  # First singular value

        # Check signal ratio is present
        idx = header.index("signal_ratio")
        assert float(data[idx]) > 0

    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)


if __name__ == "__main__":
    pytest.main([__file__])
