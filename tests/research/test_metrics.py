import pytest
import torch
import sys
import os

# Ensure the library under test is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lewm.metrics import MetricsCallback


def test_soft_rank_isotropic():
    """Identity-like matrices should result in high soft_rank."""
    # Create an identity matrix (64, 64)
    x = torch.eye(64)

    rank = MetricsCallback.compute_soft_rank(x)

    # For a perfect identity matrix, soft_rank should be close to 64
    # (Accounting for numerical epsilon)
    assert rank > 60.0


def test_soft_rank_collapsed():
    """Zero or constant matrices should result in minimal soft_rank."""
    x = torch.ones(100, 64) * 0.5
    rank = MetricsCallback.compute_soft_rank(x)

    # A single point in space has an effective dimensionality of 1.0
    assert rank < 2.0


def test_soft_rank_scaling_invariance():
    """SoftRank should be invariant to absolute scale."""
    x1 = torch.randn(100, 64)
    x2 = x1 * 1000.0

    rank1 = MetricsCallback.compute_soft_rank(x1)
    rank2 = MetricsCallback.compute_soft_rank(x2)

    assert abs(rank1 - rank2) < 0.1


def test_soft_rank_mean_centering_robustness():
    """Verify that mean-centering correctly masks constant spatial offsets."""
    # Create data that is purely 1-dimensional, but shifted far from the origin
    # [[1000, 1000], [1001, 1001], ...]
    t = torch.linspace(0, 1, 100).unsqueeze(1)
    x = torch.cat([t, t], dim=1) + 1000.0

    rank = MetricsCallback.compute_soft_rank(x)

    # This data is 1D (a line).
    # Without mean-centering, the origin-relative SVD might see it as 2D due to the large offset.
    assert 0.9 < rank < 1.1
