# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import argparse
import shutil
import json
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def fast_merge_chunks(root_dir, chunk_names, target_repo_id):
    target_path = os.path.join(root_dir, target_repo_id)
    if os.path.exists(target_path):
        print(f"🚨 Target dataset already exists at {target_path}.")
        print("Please delete or move it if you want a clean merge.")
        return

    # Check that all chunks exist
    for chunk_name in chunk_names:
        chunk_path = os.path.join(root_dir, chunk_name)
        if not os.path.exists(chunk_path):
            print(f"❌ Error: Source chunk not found at {chunk_path}.")
            return

    # Make target directories
    print("📁 Creating target directory structure...")
    os.makedirs(os.path.join(target_path, "meta"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(target_path, "videos"), exist_ok=True)

    cumulative_episodes = 0
    cumulative_frames = 0
    episodes_dfs = []
    rewards_dfs = []

    # Map each chunk's index to target indices
    for idx, chunk_name in enumerate(chunk_names):
        chunk_path = os.path.join(root_dir, chunk_name)
        print(f"\n📦 Merging chunk: {chunk_name} (Chunk index {idx})")

        # 1. Load episodes metadata
        ep_file = os.path.join(
            chunk_path, "meta", "episodes", "chunk-000", "file-000.parquet"
        )
        if not os.path.exists(ep_file):
            # Fallback to general scan if path differs
            matches = glob.glob(
                os.path.join(chunk_path, "meta", "episodes", "*", "*.parquet")
            )
            if matches:
                ep_file = matches[0]
            else:
                raise FileNotFoundError(
                    f"No episodes metadata parquet found for {chunk_name}"
                )

        ep_df = pd.read_parquet(ep_file)

        # 2. Copy the video files by shifting the chunk directories to prevent collision
        # Each chunk's videos/observation.images.xxx/chunk-000/ -> target videos/observation.images.xxx/chunk-00{idx}/
        video_src_root = os.path.join(chunk_path, "videos")
        if os.path.exists(video_src_root):
            for cam_dir in os.listdir(video_src_root):
                src_cam_path = os.path.join(video_src_root, cam_dir)
                if not os.path.isdir(src_cam_path):
                    continue
                # We expect videos inside chunk-000
                src_chunk_path = os.path.join(src_cam_path, "chunk-000")
                if os.path.exists(src_chunk_path):
                    dst_chunk_path = os.path.join(
                        target_path, "videos", cam_dir, f"chunk-{idx:03d}"
                    )
                    os.makedirs(os.path.dirname(dst_chunk_path), exist_ok=True)
                    print(f"   🎥 Copying videos for {cam_dir}...")
                    shutil.copytree(src_chunk_path, dst_chunk_path)

        # 3. Copy the data parquet files by shifting chunk directories
        # data/chunk-000/ -> data/chunk-00{idx}/
        data_src_chunk = os.path.join(chunk_path, "data", "chunk-000")
        data_dst_chunk = os.path.join(target_path, "data", f"chunk-{idx:03d}")
        if os.path.exists(data_src_chunk):
            os.makedirs(os.path.dirname(data_dst_chunk), exist_ok=True)
            print("   📄 Copying data parquet tables...")
            shutil.copytree(data_src_chunk, data_dst_chunk)

            # 4. Modify the copied parquet files to shift index & episode_index, and set chunk_index to target index
            parquet_files = glob.glob(os.path.join(data_dst_chunk, "*.parquet"))
            for p_file in tqdm(parquet_files, desc="      Updating data files"):
                df = pd.read_parquet(p_file)
                df["episode_index"] = df["episode_index"] + cumulative_episodes
                df["index"] = df["index"] + cumulative_frames
                if "data/chunk_index" in df.columns:
                    df["data/chunk_index"] = idx
                df.to_parquet(p_file)

        # 5. Shift and update episodes metadata DataFrame
        ep_df["episode_index"] = ep_df["episode_index"] + cumulative_episodes
        ep_df["dataset_from_index"] = ep_df["dataset_from_index"] + cumulative_frames
        ep_df["dataset_to_index"] = ep_df["dataset_to_index"] + cumulative_frames
        ep_df["data/chunk_index"] = idx

        # Update video chunk indices mapping in the metadata
        for col in ep_df.columns:
            if col.startswith("videos/") and col.endswith("/chunk_index"):
                ep_df[col] = idx

        episodes_dfs.append(ep_df)

        # 6. Read and shift rewards sidecar
        rewards_file = os.path.join(chunk_path, "progress_sparse.parquet")
        if os.path.exists(rewards_file):
            rw_df = pd.read_parquet(rewards_file)
            rw_df["episode_index"] = rw_df["episode_index"] + cumulative_episodes
            rw_df["index"] = rw_df["index"] + cumulative_frames
            rewards_dfs.append(rw_df)

        # Increment offsets
        # The number of episodes in this chunk is the length of the metadata DataFrame
        cumulative_episodes += len(ep_df)
        # The total frames added is the sum of lengths in this chunk
        cumulative_frames += int(ep_df["length"].sum())

    # 7. Write combined episodes metadata
    if episodes_dfs:
        merged_episodes_df = pd.concat(episodes_dfs, ignore_index=True)
        meta_ep_dir = os.path.join(target_path, "meta", "episodes", "chunk-000")
        os.makedirs(meta_ep_dir, exist_ok=True)
        merged_episodes_df.to_parquet(os.path.join(meta_ep_dir, "file-000.parquet"))
        print("\n✅ Combined episodes metadata written successfully.")

    # 8. Write combined rewards sidecar
    if rewards_dfs:
        merged_rewards_df = pd.concat(rewards_dfs, ignore_index=True)
        sidecar_path = os.path.join(target_path, "progress_sparse.parquet")
        merged_rewards_df.to_parquet(sidecar_path)
        print(
            f"✅ Combined rewards sidecar written successfully with {len(merged_rewards_df)} entries."
        )

    # 9. Copy tasks template metadata
    tasks_src = os.path.join(root_dir, chunk_names[0], "meta", "tasks.parquet")
    if os.path.exists(tasks_src):
        shutil.copy(tasks_src, os.path.join(target_path, "meta", "tasks.parquet"))

    # 10. Copy and update info.json
    info_src = os.path.join(root_dir, chunk_names[0], "meta", "info.json")
    if os.path.exists(info_src):
        with open(info_src, "r") as f:
            info = json.load(f)
        info["total_episodes"] = cumulative_episodes
        info["total_frames"] = cumulative_frames
        info["splits"] = {"train": [0, cumulative_episodes]}
        with open(os.path.join(target_path, "meta", "info.json"), "w") as f:
            json.dump(info, f, indent=4)
        print("✅ info.json updated.")

    # 11. Copy stats.json template
    stats_src = os.path.join(root_dir, chunk_names[0], "meta", "stats.json")
    if os.path.exists(stats_src):
        shutil.copy(stats_src, os.path.join(target_path, "meta", "stats.json"))
        print("✅ stats.json copied.")

    print(
        f"\n🎉 Fast Merge Complete! Unified dataset '{target_repo_id}' generated successfully in seconds."
    )
    print(
        f"📊 Total Episodes: {cumulative_episodes} | Total Frames: {cumulative_frames}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast LeRobot dataset chunk merger")
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
    print(f"Fast merging chunks {args.chunks} in {abs_root} -> {args.target}")
    fast_merge_chunks(abs_root, args.chunks, args.target)
