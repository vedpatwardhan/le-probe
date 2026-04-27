import torch
import os
import pytest
from lewm.metrics import MetricsCallback

# Standard path for pre-trained weights
CKPT_PATH = "research/checkpoints/weights.pt"


def test_pretrained_weight_ranks():
    """
    Verifies that pre-trained weights (Gold Standard) maintain high rank.
    This test asserts that mature models do NOT suffer from manifold collapse.
    """
    if not os.path.exists(CKPT_PATH):
        pytest.skip(
            f"Pre-trained weights not found at {CKPT_PATH}. Skipping benchmark."
        )

    state_dict = torch.load(CKPT_PATH, map_location="cpu")

    # Track results for summary
    rank_results = []

    for key, weight in state_dict.items():
        if "weight" in key and len(weight.shape) == 2:
            # Only analyze major layers (Transformer/MLP)
            if weight.shape[0] < 16 or weight.shape[1] < 16:
                continue

            diagnostics = MetricsCallback.compute_latent_diagnostics(weight.float())
            rank_results.append(diagnostics["soft_rank"])

            # GOLD STANDARD ASSERTION:
            # Pre-trained layers should almost ALWAYS have high spectral entropy.
            # Only specialized bottleneck layers should ever fall below 10.
            assert (
                diagnostics["soft_rank"] > 5.0
            ), f"Layer {key} has abnormally low rank: {diagnostics['soft_rank']}"

    # Overall manifold health assertion
    # average SoftRank for lewm-cube is typically >> 50
    avg_rank = sum(rank_results) / len(rank_results)
    assert avg_rank > 30.0, f"Average manifold rank is too low: {avg_rank}"


if __name__ == "__main__":
    # Allow running as a standalone script for detailed per-layer printout
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Checkpoint not found at {CKPT_PATH}")
    else:
        print(f"📂 Analyzing weights: {CKPT_PATH}")
        sd = torch.load(CKPT_PATH, map_location="cpu")
        print(f"{'Layer':<60} | {'SoftRank':<10} | {'PR_Rank':<10}")
        print("-" * 88)
        for k, w in sd.items():
            if "weight" in k and len(w.shape) == 2 and w.shape[0] >= 16:
                d = MetricsCallback.compute_latent_diagnostics(w.float())
                print(
                    f"{k[:60]:<60} | {d['soft_rank']:<10.2f} | {d['participation_ratio']:<10.2f}"
                )
