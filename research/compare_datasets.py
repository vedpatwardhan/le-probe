import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import hf_hub_download
import pandas as pd

def compare_datasets():
    print("🧪 Starting Dataset Comparison: OGBench (Cube) vs. GR-1 (Pickup)")

    # 1. Download/Load OGBench Cube Dataset (HDF5)
    print("\n📂 Loading OGBench Cube...")
    # Matches config/train/data/ogb.yaml: ogbench/cube_single_expert
    ogb_path = hf_hub_download(repo_id="quentinll/lewm", filename="ogbench/cube_single_expert.h5", repo_type="dataset")
    
    with h5py.File(ogb_path, 'r') as f:
        # OGBench typical keys: actions, observations, terminals, rewards
        ogb_actions = np.array(f['action'][:1000]) # Sample 1000 steps
        ogb_pixels = np.array(f['pixels'][:100])  # Sample 100 images
        print(f"OGB Action Shape: {ogb_actions.shape}")
        print(f"OGB Pixel Shape: {ogb_pixels.shape}")

    # 2. Load GR-1 Dataset (LeRobot)
    print("\n📂 Loading GR-1 LeRobot Dataset...")
    gr1_dataset = LeRobotDataset("vedpatwardhan/gr1_pickup_large")
    
    # Sample a few batches
    gr1_batch = gr1_dataset[0:1000]
    gr1_actions = gr1_batch['action'].numpy()
    gr1_pixels = gr1_batch['observation.images.world_center'].numpy()
    print(f"GR-1 Action Shape: {gr1_actions.shape}")
    print(f"GR-1 Pixel Shape: {gr1_pixels.shape}")

    # 3. Comparison Metrics
    comparison = {
        "Metric": ["Action Dim", "Action Mean", "Action Std", "Action Min", "Action Max", "Pixel Range"],
        "OGB-Cube": [
            ogb_actions.shape[-1],
            np.mean(ogb_actions),
            np.std(ogb_actions),
            np.min(ogb_actions),
            np.max(ogb_actions),
            f"{np.min(ogb_pixels)} to {np.max(ogb_pixels)}"
        ],
        "GR-1-Pickup": [
            gr1_actions.shape[-1],
            np.mean(gr1_actions),
            np.std(gr1_actions),
            np.min(gr1_actions),
            np.max(gr1_actions),
            f"{np.min(gr1_pixels)} to {np.max(gr1_pixels)}"
        ]
    }

    df = pd.DataFrame(comparison)
    print("\n📊 Dataset Statistics Comparison:")
    print(df.to_markdown())

    # 4. Visualization: Action Distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(ogb_actions.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title("OGB-Cube Action Distribution")
    plt.xlabel("Value")
    
    plt.subplot(1, 2, 2)
    plt.hist(gr1_actions.flatten(), bins=50, color='green', alpha=0.7)
    plt.title("GR-1-Pickup Action Distribution")
    plt.xlabel("Value")
    
    plt.tight_layout()
    plt.savefig("dataset_comparison.png")
    print("\n📈 Plot saved to dataset_comparison.png")

if __name__ == "__main__":
    compare_datasets()
