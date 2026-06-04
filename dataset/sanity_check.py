# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

import argparse
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def run_sanity_check(root_dir, repo_id):
    dataset_path = os.path.join(root_dir, repo_id)
    print(f"🔎 Running sanity check on: {dataset_path}\n")

    # 1. Check metadata files presence
    required_files = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "progress_sparse.parquet",
    ]
    for f in required_files:
        path = os.path.join(dataset_path, f)
        exists = os.path.exists(path)
        print(f"[{'✅' if exists else '❌'}] File: {f}")
        if not exists:
            print("🚨 Missing critical file! Sanity check failed.")
            return

    # 2. Verify LeRobotDataset instantiation
    try:
        print("\n🤖 Instantiating LeRobotDataset...")
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        print("✅ LeRobotDataset instantiated successfully.")
        print(f"   - Total episodes: {dataset.num_episodes} (Expected: 2000)")
        print(f"   - Total frames: {dataset.num_frames} (Expected: 64000)")

        # Verify counts
        assert (
            dataset.num_episodes == 2000
        ), f"Expected 2000 episodes, found {dataset.num_episodes}"
        assert (
            dataset.num_frames == 64000
        ), f"Expected 64000 frames, found {dataset.num_frames}"
    except Exception as e:
        print(f"❌ Error instantiating dataset: {e}")
        return

    # 3. Check Frame Fetching at Chunk Boundaries
    print("\n📸 Testing frame access at chunk boundaries...")
    # 4 chunks of 500 episodes * 32 steps = 16000 frames each
    boundaries = [0, 15999, 16000, 31999, 32000, 47999, 48000, 63999]
    all_ok = True
    for idx in boundaries:
        try:
            item = dataset[idx]
            task = item["task"]
            state_shape = item["observation.state"].shape
            print(
                f"   - Frame {idx:05d}: OK (Task: '{task}', State shape: {state_shape})"
            )
        except Exception as e:
            print(f"   - Frame {idx:05d}: ❌ FAILED - {e}")
            all_ok = False

    if all_ok:
        print("✅ All boundary frame queries succeeded.")
    else:
        print("❌ Frame query failure detected.")

    # 4. Validate rewards sidecar
    print("\n🏆 Validating reward sidecar index alignment...")
    try:
        sidecar_path = os.path.join(dataset_path, "progress_sparse.parquet")
        rw_df = pd.read_parquet(sidecar_path)

        # Verify rows count
        print(f"   - Sidecar reward rows: {len(rw_df)} (Expected: 64000)")
        assert len(rw_df) == 64000, f"Expected 64000 rows, found {len(rw_df)}"

        # Verify uniqueness of indices
        unique_indices = rw_df["index"].nunique()
        print(f"   - Unique indices: {unique_indices} (Expected: 64000)")
        assert unique_indices == 64000, f"Duplicate indices found in rewards parquet!"

        # Check index range
        min_idx, max_idx = rw_df["index"].min(), rw_df["index"].max()
        print(f"   - Index range: [{min_idx}, {max_idx}] (Expected: [0, 63999])")
        assert min_idx == 0 and max_idx == 63999, f"Index range mismatch!"

        # Check episode range
        min_ep, max_ep = rw_df["episode_index"].min(), rw_df["episode_index"].max()
        print(f"   - Episode index range: [{min_ep}, {max_ep}] (Expected: [0, 1999])")
        assert min_ep == 0 and max_ep == 1999, f"Episode index range mismatch!"

        print("✅ Reward sidecar matches the dataset frame index mapping perfectly.")
    except Exception as e:
        print(f"❌ Reward sidecar validation failed: {e}")
        return

    print("\n🎉 SANITY CHECK PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check LeRobot dataset")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="le-probe/datasets",
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="gr1_pickup_grasp_2k",
        help="Repository ID to check",
    )
    args = parser.parse_args()

    run_sanity_check(os.path.abspath(args.root_dir), args.target)
