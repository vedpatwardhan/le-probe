import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project Imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
lewm_root = os.path.join(project_root, "le_wm")

if project_root not in sys.path:
    sys.path.append(project_root)
if script_dir not in sys.path:
    sys.path.append(script_dir)
if lewm_root not in sys.path:
    sys.path.append(lewm_root)

from goal_mapper import GoalMapper


def compare_models(naive_ckpt, tuned_ckpt, json_files):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Comparing on device: {device}")

    # 1. Load Both Models
    print(f"🏗️  Loading Naive Model: {naive_ckpt}")
    mapper_naive = GoalMapper(model_path=naive_ckpt, dataset_root=".")
    model_naive = mapper_naive.model.to(device).eval()

    print(f"🏗️  Loading Tuned Model: {tuned_ckpt}")
    mapper_tuned = GoalMapper(model_path=tuned_ckpt, dataset_root=".")
    model_tuned = mapper_tuned.model.to(device).eval()

    # 2. Files already provided in signature
    print(f"📦 Auditing {len(json_files)} snapshots.")

    results = []

    # 3. Batch Processing (to be faster)
    for json_file in tqdm(json_files, desc="Auditing Snapshots"):
        with open(json_file, "r") as f:
            data = json.load(f)

        gt_reward = float(data.get("progress", 0.0))
        img_list = data["observation.images.world_center"]
        img_np = np.array(img_list, dtype=np.uint8).transpose(1, 2, 0)

        # Transform (same for both)
        batch = mapper_naive.transform({"pixels": img_np})
        pixel_tensor = (
            batch["pixels"].to(device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, C, H, W)

        with torch.no_grad():
            # Naive Pred
            info_n = model_naive.encode({"pixels": pixel_tensor})
            pred_n = model_naive.reward_head(info_n["emb"]).item()

            # Tuned Pred
            info_t = model_tuned.encode({"pixels": pixel_tensor})
            pred_t = model_tuned.reward_head(info_t["emb"]).item()

        results.append({"gt": gt_reward, "naive": pred_n, "tuned": pred_t})

    # 4. Calculate Stats
    gt_arr = np.array([r["gt"] for r in results])
    naive_arr = np.array([r["naive"] for r in results])
    tuned_arr = np.array([r["tuned"] for r in results])

    naive_mse = np.mean((gt_arr - naive_arr) ** 2)
    tuned_mse = np.mean((gt_arr - tuned_arr) ** 2)

    naive_mae = np.mean(np.abs(gt_arr - naive_arr))
    tuned_mae = np.mean(np.abs(gt_arr - tuned_arr))

    print("\n" + "=" * 40)
    print("📊 GLOBAL AUDIT RESULTS")
    print("=" * 40)
    print(f"NAIVE MODEL (Original):")
    print(f"  - MSE: {naive_mse:.4f}")
    print(f"  - MAE: {naive_mae:.4f}")
    print(f"TUNED MODEL (Calibrated):")
    print(f"  - MSE: {tuned_mse:.4f}")
    print(f"  - MAE: {tuned_mae:.4f}")
    print("=" * 40)
    print(f"🔥 Error Reduction: {((naive_mse - tuned_mse)/naive_mse)*100:.2f}%")
    print("=" * 40)

    # 5. Generate Comparison Plots
    plt.figure(figsize=(15, 10))

    # Scatter Plots
    plt.subplot(2, 2, 1)
    plt.scatter(gt_arr, naive_arr, alpha=0.3, color="red")
    plt.plot([0, 10], [0, 10], "k--", alpha=0.5)
    plt.title("Naive Model: GT vs Pred (Scatter)")
    plt.xlabel("Ground Truth Reward")
    plt.ylabel("Predicted Reward")

    plt.subplot(2, 2, 2)
    plt.scatter(gt_arr, tuned_arr, alpha=0.3, color="green")
    plt.plot([0, 10], [0, 10], "k--", alpha=0.5)
    plt.title("Tuned Model: GT vs Pred (Scatter)")
    plt.xlabel("Ground Truth Reward")
    plt.ylabel("Predicted Reward")

    # Distribution Plots
    plt.subplot(2, 1, 2)
    bins = np.linspace(-1, 11, 50)
    plt.hist(gt_arr, bins=bins, alpha=0.3, label="Ground Truth", color="blue")
    plt.hist(naive_arr, bins=bins, alpha=0.3, label="Naive Pred", color="red")
    plt.hist(tuned_arr, bins=bins, alpha=0.3, label="Tuned Pred", color="green")
    plt.title("Reward Distributions (GT vs Naive vs Tuned)")
    plt.xlabel("Reward Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reward_comparison_audit.png")
    print("\n📈 Comprehensive audit plot saved to reward_comparison_audit.png")


if __name__ == "__main__":
    # Combined Audit: Failures + Successes
    snapshot_dirs = [
        "cortex-gr1/datasets/vedpatwardhan/gr1_reward_pred",  # 1000 Failures
        "cortex-gr1/datasets/vedpatwardhan/gr1_success_gold",  # 200 Successes
    ]

    # Collect all JSON files from all specified directories
    all_json_files = []
    for s_dir in snapshot_dirs:
        path = Path(s_dir)
        if path.exists():
            all_json_files.extend(list(path.glob("*.json")))

    compare_models(
        naive_ckpt="gr1-epoch=99-step=004400.ckpt",
        tuned_ckpt="gr1_reward_tuned.ckpt",
        json_files=all_json_files,
    )
