# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_snapshots(snapshots_dir):
    snap_path = Path(snapshots_dir)
    json_files = sorted([f for f in snap_path.glob("*.json")])

    if not json_files:
        print(f"No snapshots found in {snapshots_dir}")
        return

    num_snaps = len(json_files)
    cols = 4
    rows = (num_snaps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract Image (C, H, W) -> (H, W, C)
        img_data = np.array(data["observation.images.world_center"], dtype=np.uint8)
        img_data = img_data.transpose(1, 2, 0)

        # Extract Reward
        reward = data.get("progress", 0.0)

        ax = axes[i]
        ax.imshow(img_data)
        ax.set_title(f"ID: {json_file.stem}\nReward: {reward:.2f}")
        ax.axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_path = "snapshot_visualization.png"
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    visualize_snapshots("le-probe/snapshots")
