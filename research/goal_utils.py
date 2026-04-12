"""
GOAL EXTRACTION UTILITIES (The "Archeologist")
Role: Extraction of visual success frames from the dataset.

This module provides standalone helper functions to search through the
pickup dataset and retrieve the "Success" (last) frame of an episode.
It abstracts away the complexities of Parquet searching and OpenCV video
decoding.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import cv2


def get_goal_pixels(dataset_root, episode_idx=0, img_size=224):
    """
    Fetches the SUCCESS frame (the last frame) of a specific episode.
    By definition, the last frame of your pickup episodes is the successful lift.
    """
    dataset_root = Path(dataset_root)
    meta_dir = dataset_root / "data"

    # 1. Find the video and metadata for this episode
    # We look for the parquet that contains this episode_index
    found_info = None
    for pq in sorted(meta_dir.glob("**/*.parquet")):
        df = pd.read_parquet(pq)
        ep_data = df[df["episode_index"] == episode_idx]
        if not ep_data.empty:
            last_frame_idx = ep_data["frame_index"].max()
            video_path = (
                dataset_root
                / "videos"
                / "observation.images.world_center"
                / "chunk-000"
                / f"file-{episode_idx:03d}.mp4"
            )
            if video_path.exists():
                found_info = {
                    "video_path": str(video_path),
                    "frame_idx": last_frame_idx,
                }
                break

    if not found_info:
        print(f"❌ Could not find episode {episode_idx} in {dataset_root}")
        return None

    print(f"🎯 Goal established: Last frame of Episode {episode_idx}")

    # 2. Extract frame via OpenCV
    cap = cv2.VideoCapture(found_info["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, found_info["frame_idx"])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ OpenCV failed to read success frame.")
        return None

    # 3. Transform for JEPA (BGR -> RGB -> Tensor)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=(img_size, img_size)),
        ]
    )

    return transform(frame).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)
