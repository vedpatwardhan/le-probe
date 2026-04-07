import h5py
import hdf5plugin  # Required to read zstd-compressed HDF5 files
import numpy as np
import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import hf_hub_download
import pandas as pd
import os


def compare_datasets():
    print("🧪 Starting Dataset Comparison: PushT vs. GR-1")

    # 1. Download and Extract PushT Dataset (h5.zst)
    ogb_path = "pusht_expert_train.h5"
    if not os.path.exists(ogb_path):
        print("\n📦 Fetching PushT Dataset Archive...")
        archive_path = hf_hub_download(
            repo_id="quentinll/lewm-pusht",
            filename="pusht_expert_train.h5.zst",
            repo_type="dataset",
        )

        print(f"📦 Decompressing {archive_path}...")
        import subprocess

        # Use zstd -d to decompress the .h5.zst file
        subprocess.run(["zstd", "-d", archive_path, "-o", ogb_path, "-f"], check=True)
    else:
        print(f"\n✅ Found existing dataset at {ogb_path}, skipping download.")

    with h5py.File(ogb_path, "r") as f:
        # PushT keys: action, pixels, proprio, state
        ogb_actions = np.array(f["action"][:1000])
        ogb_pixels = np.array(f["pixels"][:100])
        ogb_state = np.array(f["proprio"][:1000])
        print(f"PushT Action Shape: {ogb_actions.shape}")
        print(f"PushT State Shape: {ogb_state.shape}")

    # 2. Load GR-1 Dataset (LeRobot)
    print("\n📂 Loading GR-1 LeRobot Dataset...")
    gr1_dataset = LeRobotDataset("vedpatwardhan/gr1_pickup_large")

    num_samples = min(1000, len(gr1_dataset))
    print(f"📥 Sampling {num_samples} frames from GR-1...")

    gr1_actions_list = []
    gr1_pixels_list = []
    gr1_state_list = []

    for i in range(num_samples):
        frame = gr1_dataset[i]
        gr1_actions_list.append(frame["action"])
        gr1_pixels_list.append(frame["observation.images.world_center"])
        gr1_state_list.append(frame["observation.state"])

    gr1_actions = torch.stack(gr1_actions_list).numpy()
    gr1_pixels = torch.stack(gr1_pixels_list).numpy()
    gr1_state = torch.stack(gr1_state_list).numpy()

    print(f"GR-1 Action Shape: {gr1_actions.shape}")
    print(f"GR-1 State Shape: {gr1_state.shape}")

    # 3. Comparison Metrics
    comparison = {
        "Metric": [
            "Action Dim",
            "Action Std",
            "Action Range",
            "State Dim",
            "State Std",
            "State Range",
            "Pixel Range",
        ],
        "PushT": [
            ogb_actions.shape[-1],
            np.std(ogb_actions),
            f"[{np.min(ogb_actions):.2f}, {np.max(ogb_actions):.2f}]",
            ogb_state.shape[-1],
            np.std(ogb_state),
            f"[{np.min(ogb_state):.2f}, {np.max(ogb_state):.2f}]",
            f"[{np.min(ogb_pixels)}, {np.max(ogb_pixels)}]",
        ],
        "GR-1-Pickup": [
            gr1_actions.shape[-1],
            np.std(gr1_actions),
            f"[{np.min(gr1_actions):.2f}, {np.max(gr1_actions):.2f}]",
            gr1_state.shape[-1],
            np.std(gr1_state),
            f"[{np.min(gr1_state):.2f}, {np.max(gr1_state):.2f}]",
            f"[{np.min(gr1_pixels)}, {np.max(gr1_pixels)}]",
        ],
    }

    df = pd.DataFrame(comparison)
    print("\n📊 Dataset Statistics Comparison:")
    print(df.to_markdown())

    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Action Dist
    axes[0].hist(ogb_actions.flatten(), bins=50, color="blue", alpha=0.5, label="PushT")
    axes[0].hist(gr1_actions.flatten(), bins=50, color="green", alpha=0.5, label="GR-1")
    axes[0].set_title("Action Distributions")
    axes[0].legend()

    # State Dist
    axes[1].hist(ogb_state.flatten(), bins=50, color="blue", alpha=0.5, label="PushT")
    axes[1].hist(gr1_state.flatten(), bins=50, color="green", alpha=0.5, label="GR-1")
    axes[1].set_title("State (Proprio) Distributions")
    axes[1].legend()

    # Pixel Dist
    axes[2].hist(ogb_pixels.flatten(), bins=50, color="blue", alpha=0.5, label="PushT")
    axes[2].hist(gr1_pixels.flatten(), bins=50, color="green", alpha=0.5, label="GR-1")
    axes[2].set_title("Pixel Distributions")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("dataset_comparison.png")
    print("\n📈 Plot saved to dataset_comparison.png")


if __name__ == "__main__":
    compare_datasets()
