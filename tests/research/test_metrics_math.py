import torch
import pytest
from lewm.metrics import MetricsCallback

# Constants for synthetic data
B, D = 128, 192


def test_isotropic_gaussian_rank():
    """Case A: Isotropic Gaussian should have high rank (close to D)."""
    torch.manual_seed(42)
    z = torch.randn(B, D)
    metrics = MetricsCallback.compute_latent_diagnostics(z)

    # In 128x192, SoftRank is typically > 100, PR > 70
    assert metrics["soft_rank"] > 100.0
    assert metrics["participation_ratio"] > 70.0
    assert metrics["soft_rank"] >= metrics["participation_ratio"]


def test_partial_manifold_rank():
    """Case B: 10D Manifold should have rank near 10."""
    torch.manual_seed(42)
    z = torch.zeros(B, D)
    z[:, :10] = torch.randn(B, 10)
    metrics = MetricsCallback.compute_latent_diagnostics(z)

    # Rank should be close to the actual DOF (10)
    assert 8.0 < metrics["soft_rank"] < 12.0
    assert 8.0 < metrics["participation_ratio"] < 12.0


def test_spike_manifold_rank():
    """Case C: A single huge variance dimension (spike) should suppress rank."""
    torch.manual_seed(42)
    z = torch.randn(B, D) * 0.1  # Tiny noise
    z[:, :5] = torch.randn(B, 5) * 5.0  # Moderate variance
    z[:, 0] = torch.randn(B) * 100.0  # MASSIVE spike

    metrics = MetricsCallback.compute_latent_diagnostics(z)

    # Participation Ratio is very sensitive to spikes (should be ~1-2)
    # SoftRank (Entropy) is more robust but still lowered
    assert metrics["participation_ratio"] < 5.0
    assert metrics["soft_rank"] < 20.0


def test_collapsed_manifold_rank():
    """Case D: Perfectly collapsed data should return 1.0."""
    z = torch.zeros(B, D)
    metrics = MetricsCallback.compute_latent_diagnostics(z)

    assert metrics["soft_rank"] == 1.0
    assert metrics["participation_ratio"] == 1.0


def test_single_sample_safety():
    """Edge Case: Batch size 1 should return 1.0 gracefully."""
    z = torch.randn(1, D)
    metrics = MetricsCallback.compute_latent_diagnostics(z)

    assert metrics["soft_rank"] == 1.0
    assert metrics["participation_ratio"] == 1.0


def test_numerical_stability_low_variance():
    """Edge Case: Extremely low variance should not trigger NaNs."""
    z = torch.randn(B, D) * 1e-20
    metrics = MetricsCallback.compute_latent_diagnostics(z)

    assert metrics["soft_rank"] == 1.0
    assert metrics["participation_ratio"] == 1.0
