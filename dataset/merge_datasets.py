# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import argparse
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def merge_chunks(root_dir, chunk_names, target_repo_id):
    # Determine target dataset path
    target_path = os.path.join(root_dir, target_repo_id)
    if os.path.exists(target_path):
        print(f"🚨 Target dataset already exists at {target_path}.")
        print("Please delete or move it if you want a clean merge.")
        return

    # Load the first chunk to inspect/replicate its configuration features
    first_chunk_path = os.path.join(root_dir, chunk_names[0])
    if not os.path.exists(first_chunk_path):
        print(
            f"❌ Error: First chunk not found at {first_chunk_path}. Cannot replicate features."
        )
        return

    print(f"📖 Probing features from first chunk: {chunk_names[0]}")
    first_ds = LeRobotDataset(repo_id=chunk_names[0], root=first_chunk_path)

    # Create the target dataset
    target_ds = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=first_ds.fps,
        root=target_path,
        features=first_ds.features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        video_backend="ffmpeg",
        vcodec="h264",
    )
    print(f"✨ Created target dataset at: {target_path}")

    all_rewards = []

    # Loop over chunks
    for chunk_name in chunk_names:
        chunk_path = os.path.join(root_dir, chunk_name)
        if not os.path.exists(chunk_path):
            print(
                f"⚠️ Warning: Chunk {chunk_name} not found at {chunk_path}. Skipping."
            )
            continue

        print(f"📦 Merging chunk: {chunk_name}...")
        chunk_ds = LeRobotDataset(repo_id=chunk_name, root=chunk_path)

        # Load the reward sidecar for this chunk if it exists
        reward_sidecar_path = os.path.join(chunk_path, "progress_sparse.parquet")
        chunk_rewards_df = None
        if os.path.exists(reward_sidecar_path):
            chunk_rewards_df = pd.read_parquet(reward_sidecar_path)
            print(f"   ℹ️ Loaded rewards sidecar from {reward_sidecar_path}")

        # Iterate over all episodes of chunk_ds
        for ep_idx in tqdm(
            range(chunk_ds.num_episodes), desc=f"Episodes in {chunk_name}"
        ):
            start_frame, end_frame = chunk_ds.episode_bounds[ep_idx]
            task = chunk_ds.hf_dataset[start_frame]["task"]

            for frame_idx in range(start_frame, end_frame):
                frame_data = chunk_ds[frame_idx]

                # Reconstruct dict expected by add_frame
                add_data = {}
                for key in first_ds.features:
                    val = frame_data[key]
                    # Transpose (C, H, W) images/video tensors to (H, W, C) numpy or PIL for image writer compatibility
                    if first_ds.features[key]["dtype"] in ["image", "video"]:
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().numpy()
                        # If shape is channel-first (e.g. 3, 224, 224), transpose to channel-last
                        if val.ndim == 3 and val.shape[0] in [1, 3]:
                            val = np.transpose(val, (1, 2, 0))
                        val = Image.fromarray(
                            (val * 255).astype(np.uint8)
                            if val.dtype == np.float32 or val.max() <= 1.0
                            else val.astype(np.uint8)
                        )
                    elif isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    add_data[key] = val

                add_data["task"] = task
                target_ds.add_frame(add_data)

            target_ds.save_episode(parallel_encoding=False)

            # Map the rewards sidecar indices
            target_episode_idx = target_ds.num_episodes - 1
            target_start_frame = target_ds.num_frames - (end_frame - start_frame)

            if chunk_rewards_df is not None:
                ep_rows = chunk_rewards_df[
                    chunk_rewards_df["episode_index"] == ep_idx
                ].copy()
                if not ep_rows.empty:
                    ep_rows = ep_rows.sort_values("index")
                    for i, (_, row) in enumerate(ep_rows.iterrows()):
                        all_rewards.append(
                            {
                                "index": target_start_frame + i,
                                "episode_index": target_episode_idx,
                                "progress_sparse": row["progress_sparse"],
                                "progress_dense": row.get(
                                    "progress_dense", row["progress_sparse"]
                                ),
                                "value": row.get("value", 0.0),
                            }
                        )

    # Save target rewards sidecar
    if all_rewards:
        reward_df = pd.DataFrame(all_rewards)
        sidecar_path = os.path.join(target_path, "progress_sparse.parquet")
        reward_df.to_parquet(sidecar_path)
        print(
            f"✅ Successfully wrote rewards sidecar to: {sidecar_path} with {len(reward_df)} entries."
        )

    print(
        f"\n🎉 Merge complete! Combined dataset has {target_ds.num_episodes} episodes and {target_ds.num_frames} total frames."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LeRobot dataset chunks into a unified dataset"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="le-probe/datasets",
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=["gr1_chunk_0", "gr1_chunk_1", "gr1_chunk_2", "gr1_chunk_3"],
        help="List of chunks to merge",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="gr1_pickup_2k_hybrid",
        help="Target unified repository ID",
    )

    args = parser.parse_args()

    # Resolve absolute path for root-dir
    abs_root = os.path.abspath(args.root_dir)
    print(f"Merging chunks {args.chunks} in {abs_root} -> {args.target}")
    merge_chunks(abs_root, args.chunks, args.target)
