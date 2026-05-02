import os
import json
import cv2
import numpy as np
from pathlib import Path


def visualize_audit(report_path, dataset_dir, output_dir="feature_gallery"):
    with open(report_path, "r") as f:
        report = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Model Spec
    tokens_per_moment = 771
    tokens_per_view = 257  # 16x16 + 1 CLS
    grid_size = 16
    input_res = (224, 224)

    # Camera views mapping (Adjust these if your dataset names differ)
    camera_views = [
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_wrist",
    ]

    for fid, examples in report.items():
        print(f"🖼️ Generating gallery for Feature {fid}...")
        feat_dir = output_path / f"feature_{fid}"
        feat_dir.mkdir(exist_ok=True)

        for rank, (val, global_idx) in enumerate(examples):
            # 1. Map to Dataset
            moment_idx = global_idx // tokens_per_moment
            token_idx = global_idx % tokens_per_moment

            episode_idx = moment_idx // 32
            frame_idx = moment_idx % 32

            view_idx = token_idx // tokens_per_view
            patch_idx = token_idx % tokens_per_view

            # 2. Extract Frame from Video
            view_name = (
                camera_views[view_idx]
                if view_idx < len(camera_views)
                else camera_views[0]
            )
            video_path = (
                Path(dataset_dir) / "videos" / f"{view_name}_ep{episode_idx:06d}.mp4"
            )

            if not video_path.exists():
                # Try alternative naming
                video_path = (
                    Path(dataset_dir)
                    / "videos"
                    / view_name
                    / f"episode_{episode_idx:06d}.mp4"
                )

            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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

                        cv2.rectangle(frame, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"Act: {val:.2f}",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )

                    save_name = f"rank{rank}_ep{episode_idx}_f{frame_idx}_v{view_idx}_r{row if patch_idx > 0 else 'CLS'}.jpg"
                    cv2.imwrite(str(feat_dir / save_name), frame)
                else:
                    print(f"⚠️ Failed to read frame {frame_idx} from {video_path}")
            else:
                print(f"⚠️ Video not found: {video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="Path to audit JSON")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to LeRobot dataset"
    )
    parser.add_argument(
        "--output", type=str, default="feature_gallery", help="Output folder"
    )
    args = parser.parse_args()

    visualize_audit(args.report, args.dataset, args.output)
