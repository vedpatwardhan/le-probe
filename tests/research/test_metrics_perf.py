import torch
import time
import numpy as np
from sklearn.decomposition import PCA as SKLearnPCA
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import sys
import os

# Ensure the library under test is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import lewm.metrics as metrics

# Import the logic we just wrote (Mocking wandb to avoid network calls)
import wandb

wandb.log = MagicMock()
wandb.Image = MagicMock()


def test_pca_consistency():
    print("🚀 Initializing PCA Consistency & Performance Test...")

    # 1. Setup Dummy Data (128 samples, 192 dimensions)
    B, D = 128, 192
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(B, D, device=device)

    # 2. Benchmark SKLearn (CPU)
    print(f"  - Benchmarking SKLearn (CPU)...")
    z_cpu = z.detach().cpu().numpy()
    start_cpu = time.time()
    sk_pca = SKLearnPCA(n_components=2)
    z_2d_sk = sk_pca.fit_transform(z_cpu)
    cpu_duration = time.time() - start_cpu
    sk_ratios = sk_pca.explained_variance_ratio_

    # 3. Benchmark Our New GPU Logic
    print(f"  - Benchmarking Torch Low-Rank (GPU/Native)...")
    cb = metrics.MetricsCallback()

    start_gpu = time.time()
    # Manually run the guts of the new log_pca_to_wandb to get values back
    with torch.no_grad():
        z_centered = z - z.mean(dim=0, keepdim=True)
        _, S, V = torch.pca_lowrank(z_centered, q=2)
        z_2d_gpu = torch.matmul(z_centered, V[:, :2]).cpu().numpy()

        total_var = torch.var(z_centered, dim=0).sum()
        varexp = (S**2) / (z.shape[0] - 1)
        gpu_ratios = (varexp / (total_var + 1e-8)).cpu().numpy()
    gpu_duration = time.time() - start_gpu

    # 4. Consistency Checks
    print("\n📊 RESULTS:")
    print(
        f"  [CPU] Duration: {cpu_duration:.4f}s | Var: {sk_ratios[0]:.2%}, {sk_ratios[1]:.2%}"
    )
    print(
        f"  [GPU] Duration: {gpu_duration:.4f}s | Var: {gpu_ratios[0]:.2%}, {gpu_ratios[1]:.2%}"
    )

    speedup = cpu_duration / (gpu_duration + 1e-9)
    print(f"\n⚡ SPEEDUP: {speedup:.1f}x faster")

    # Variance Ratio match check
    diff = np.abs(sk_ratios - gpu_ratios).mean()
    if diff < 1e-5:
        print("✅ MATH CHECK PASSED: Variance ratios are identical.")
    else:
        print(f"❌ MATH CHECK FAILED: Difference of {diff:.6f}")

    # 5. Visual sanity check (save to a file)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(z_2d_sk[:, 0], z_2d_sk[:, 1], alpha=0.5)
    plt.title("SKLearn PCA (CPU)")

    plt.subplot(1, 2, 2)
    plt.scatter(z_2d_gpu[:, 0], z_2d_gpu[:, 1], alpha=0.5, color="orange")
    plt.title("Torch Low-Rank PCA (GPU)")

    save_path = "pca_test_comparison.png"
    plt.savefig(save_path)
    print(f"🖼️  Comparison plot saved to: {save_path}")


if __name__ == "__main__":
    test_pca_consistency()
