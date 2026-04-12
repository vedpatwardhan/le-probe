import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import cv2


def find_goal_pixels(target_xyz, model_path, dataset_root, img_size=224):
    """
    Standalone utility to find goal pixels using OpenCV for robustness.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = Path(dataset_root)
    meta_dir = dataset_root / "data"

    found_info = None
    target_xyz = np.array(target_xyz)

    print(f"🔍 Searching for coord {target_xyz}...")

    for chunk in sorted(meta_dir.glob("chunk-*")):
        for pq in sorted(chunk.glob("*.parquet")):
            df = pd.read_parquet(pq)
            states = np.stack(df["observation.state"].values)
            cube_pos = states[:, 32:35]

            dists = np.linalg.norm(cube_pos - target_xyz, axis=1)
            min_idx = np.argmin(dists)

            if dists[min_idx] < 0.05:
                row = df.iloc[min_idx]
                ep_idx = row["episode_index"]
                frame_idx = row["frame_index"]

                video_path = (
                    dataset_root
                    / "videos"
                    / "observation.images.world_center"
                    / "chunk-000"
                    / f"file-{ep_idx:03d}.mp4"
                )

                if video_path.exists():
                    found_info = {"video_path": str(video_path), "frame_idx": frame_idx}
                    break
        if found_info:
            break

    if not found_info:
        return None

    print(f"✅ Found frame {found_info['frame_idx']} in {found_info['video_path']}")

    # Use OpenCV instead of torchcodec
    cap = cv2.VideoCapture(found_info["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, found_info["frame_idx"])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ OpenCV failed to read frame.")
        return None

    # CV2 reads in BGR, convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=(img_size, img_size)),
        ]
    )

    pixels = transform(frame).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)
    return pixels
