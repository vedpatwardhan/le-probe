import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def harvest_spectrum_batch(dataset_repo, steps=[8, 16, 24, 31], num_episodes=200):
    dataset_path = os.path.abspath(dataset_repo)
    repo_id = os.path.basename(dataset_path)

    print(f"🚜 Harvesting reward spectrum ({len(steps)} frames/ep) from {repo_id}...")
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_path)

    # Target Folder
    output_dir = Path("cortex-gr1/datasets/vedpatwardhan/gr1_success_gold")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sidecar for rewards
    sidecar_path = os.path.join(dataset_path, "progress_sparse.parquet")
    sdf = pd.read_parquet(sidecar_path)

    harvested = 0
    for ep_idx in tqdm(range(min(num_episodes, ds.num_episodes))):
        for step in steps:
            global_idx = (ep_idx * 32) + step

            if global_idx >= ds.num_frames:
                break

            frame = ds[global_idx]
            reward = float(sdf.iloc[global_idx]["progress_sparse"])

            img = frame["observation.images.world_center"]
            img_np = np.array(img)

            if img_np.shape[-1] == 3:
                img_np = img_np.transpose(2, 0, 1)

            snapshot = {
                "observation.images.world_center": img_np.tolist(),
                "progress": reward,
                "episode_index": int(ep_idx),
                "step": int(step),
            }

            snap_path = output_dir / f"spec_{ep_idx:03d}_s{step:02d}.json"
            with open(snap_path, "w") as f:
                json.dump(snapshot, f)

            harvested += 1

    print(f"✅ Spectrum harvest complete! {harvested} snapshots stored in {output_dir}")


if __name__ == "__main__":
    harvest_spectrum_batch("cortex-gr1/datasets/vedpatwardhan/gr1_pickup_grasp")
