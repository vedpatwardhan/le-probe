"""
cache_fused_dataset.py

High-speed direct dataset pre-cache compiler. Bypasses torchcodec completely by using OpenCV
to decode, resize, and fuse RGB and Skeleton visual streams, stacking views and compiling
them along with actions and proprioceptive states directly into serialized Torch files.
"""

import os
import sys
import gzip
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
from lewm.skeleton.dino_constants import (  # noqa: E402
    validate_dino_waypoints,
    zeros_dino_waypoints,
)


def main(repo_id="gr1_pickup_grasp"):
    lerobot_home = os.environ.get("LEROBO_HOME")
    if lerobot_home:
        local_path = Path(lerobot_home) / repo_id
        if local_path.exists():
            dataset = LeRobotDataset(repo_id, root=str(local_path))
        else:
            dataset = LeRobotDataset(repo_id)
    else:
        dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)

    cache_dir = dataset_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
    total_episodes = dataset.num_episodes

    print("🚀 Compiling High-Speed Direct Fused Disk Cache...")
    print(f"🎬 Target: {total_episodes} Episodes (Center, Left, Right, Top, Wrist)")

    for ep in tqdm(range(total_episodes), desc="Caching Episodes"):
        ep_meta = dataset.meta.episodes[ep]
        c_idx = ep_meta["data/chunk_index"]
        f_idx = ep_meta["data/file_index"]

        # 1. Load Parquet for actions and states
        parquet_path = dataset_path / f"data/chunk-{c_idx:03d}/file-{f_idx:03d}.parquet"
        if not parquet_path.exists():
            print(f"⚠️ Parquet missing for Episode {ep}: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        # Extract states and actions
        state_cols = [c for c in df.columns if c.startswith("observation.state")]
        action_cols = [c for c in df.columns if c.startswith("action")]

        # Handle state extraction
        if len(state_cols) == 1:
            state_tensor = torch.from_numpy(np.stack(df[state_cols[0]].values)).float()
        else:
            state_tensor = torch.from_numpy(df[state_cols].values).float()

        # Handle action extraction
        if len(action_cols) == 1:
            action_tensor = torch.from_numpy(
                np.stack(df[action_cols[0]].values)
            ).float()
        else:
            action_tensor = torch.from_numpy(df[action_cols].values).float()

        # We will compile visual frames of shape [32, 5, 4, 224, 224]
        # 32 steps, 5 views, 4 channels (RGB + Skeleton), 224x224
        episode_pixels = []

        for view in views:
            skel_path = (
                dataset_path
                / f"videos/observation.images.{view}_tiled/chunk-{c_idx:03d}/file-{f_idx:03d}.mp4"
            )

            if not skel_path.exists():
                print(f"⚠️ Missing tiled video for Ep {ep} {view}")
                continue

            cap_skel = cv2.VideoCapture(str(skel_path))

            view_frames = []
            for frame_idx in range(32):
                ret_skel, frame_tiled = cap_skel.read()

                if not ret_skel:
                    # Fallback to zero frames if video ends prematurely
                    rgb_224 = np.zeros((224, 224, 3), dtype=np.uint8)
                    skel_224 = np.zeros((224, 224), dtype=np.uint8)
                else:
                    # The tiled video frame is BGR of shape [H, W, 3] (e.g. 224x448x3 or 480x960x3)
                    # Left half is RGB frame, right half is skeleton mask
                    h_tiled, w_tiled, _ = frame_tiled.shape

                    if w_tiled == 960:
                        frame_rgb = frame_tiled[:, :480]
                        frame_skel = cv2.cvtColor(
                            frame_tiled[:, 480:], cv2.COLOR_BGR2GRAY
                        )
                    elif w_tiled == 448:
                        frame_rgb = frame_tiled[:, :224]
                        frame_skel = cv2.cvtColor(
                            frame_tiled[:, 224:], cv2.COLOR_BGR2GRAY
                        )
                    else:
                        mid = w_tiled // 2
                        frame_rgb = frame_tiled[:, :mid]
                        frame_skel = cv2.cvtColor(
                            frame_tiled[:, mid:], cv2.COLOR_BGR2GRAY
                        )

                    # OpenCV reads in BGR, convert RGB frame to RGB format
                    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

                    # Resize to 224x224
                    rgb_224 = cv2.resize(
                        frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR
                    )
                    skel_224 = cv2.resize(
                        frame_skel, (224, 224), interpolation=cv2.INTER_AREA
                    )

                # Fuse channels: RGB (3) + Skeleton (1)
                fused = np.zeros((4, 224, 224), dtype=np.uint8)
                # Transpose RGB from [H, W, 3] -> [3, H, W]
                fused[:3] = rgb_224.transpose(2, 0, 1)
                fused[3] = skel_224

                view_frames.append(torch.from_numpy(fused))

            cap_skel.release()

            # Stack 32 frames of the view -> Shape [32, 4, 224, 224]
            view_tensor = torch.stack(view_frames, dim=0)
            episode_pixels.append(view_tensor)

        # Stack across 5 views -> Shape [32, 5, 4, 224, 224]
        stacked_pixels = torch.stack(episode_pixels, dim=1)

        # 2. Package DINO Waypoint Anchors if pre-computed
        dino_pt_path = (
            dataset_path / f"cache_dino/chunk-{c_idx:03d}/file-{f_idx:03d}_dino.pt"
        )
        if dino_pt_path.exists():
            dino_waypoints = validate_dino_waypoints(
                torch.load(dino_pt_path, map_location="cpu")
            )
        else:
            dino_waypoints = zeros_dino_waypoints()
            print(f"⚠️ DINO prior missing for Ep {ep}. Padded with zeros.")

        # Pack into serialized dict
        packaged_data = {
            "pixels": stacked_pixels,  # uint8 [32, 5, 4, 224, 224]
            "state": state_tensor,  # float [32, D_state]
            "action": action_tensor,  # float [32, D_action]
            "dino_waypoints": dino_waypoints,  # float [4, 5, 384] multi-view anchors
        }

        # Save to disk
        out_path = cache_dir / f"episode_{ep:03d}_fused.pt"
        torch.save(packaged_data, out_path)

    print(f"🎉 Pre-compiled cache successfully generated inside: {cache_dir}")


if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "gr1_pickup_grasp"
    main(repo_id)
