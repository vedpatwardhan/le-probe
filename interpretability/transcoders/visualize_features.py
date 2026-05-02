import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def visualize_audit(report_path, dataset_dir, output_dir="feature_gallery"):
    with open(report_path, "r") as f:
        report = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Model Spec (LeWM v17)
    tokens_per_moment = 771
    tokens_per_frame = 257  # 16x16 + 1 CLS
    grid_size = 16
    input_res = (224, 224)

    # Primary camera for 'pixels' input
    camera_key = "observation.images.world_center"

    for fid, examples in report.items():
        print(f"🖼️ Generating gallery for Feature {fid}...")
        feat_dir = output_path / f"feature_{fid}"
        feat_dir.mkdir(exist_ok=True)

        for rank, (val, global_idx) in enumerate(examples):
            # 1. Map to Dataset
            # 771 tokens = 3 consecutive frames (T=3) * 257 tokens/frame
            moment_idx = global_idx // tokens_per_moment
            token_idx = global_idx % tokens_per_moment

            episode_idx = moment_idx // 32
            base_frame_idx = moment_idx % 32

            # Find the specific frame in the T=3 window
            time_offset = token_idx // tokens_per_frame  # 0, 1, or 2
            patch_idx = token_idx % tokens_per_frame

            actual_frame_idx = base_frame_idx + time_offset

            # 2. Extract Frame from Video
            # Structure: videos/observation.images.world_center/episode_000000.mp4
            video_path = (
                Path(dataset_dir)
                / "videos"
                / camera_key
                / f"episode_{episode_idx:06d}.mp4"
            )

            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    # Resize to match model input (224x224)
                    frame = cv2.resize(frame, input_res)

                    # 3. Draw Highlight
                    if patch_idx > 0:
                        p_idx = patch_idx - 1
                        row = p_idx // grid_size
                        col = p_idx % grid_size

                        # Calculate patch pixel coords
                        ph, pw = input_res[0] // grid_size, input_res[1] // grid_size
                        x, y = col * pw, row * ph

                        cv2.rectangle(frame, (x, y), (x + pw, y + ph), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Act: {val:.2f}",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                        )

                    save_name = f"rank{rank}_ep{episode_idx}_f{actual_frame_idx}_t{time_offset}_{'patch' if patch_idx > 0 else 'CLS'}.jpg"
                    cv2.imwrite(str(feat_dir / save_name), frame)
                else:
                    print(
                        f"⚠️ Failed to read frame {actual_frame_idx} from {video_path}"
                    )
            else:
                print(f"⚠️ Video not found: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="Path to audit JSON")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/gr1_pickup_grasp",
        help="Local path or Target path",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="vedpatwardhan/gr1_pickup_grasp",
        help="HF Repo ID for auto-download",
    )
    parser.add_argument(
        "--output", type=str, default="feature_gallery", help="Output folder"
    )
    args = parser.parse_args()

    # Auto-download if missing
    dataset_path = Path(args.dataset)
    if not (dataset_path / "videos").exists():
        print(f"📥 Dataset missing. Downloading videos from {args.repo_id}...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(dataset_path),
            allow_patterns=["videos/*"],
        )

    visualize_audit(args.report, str(dataset_path), args.output)
