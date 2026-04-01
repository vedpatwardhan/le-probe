import os
import imageio
from pathlib import Path
import sys


def reconstruct_session(session_path):
    """Compiles all cameras in a session into MP4 videos."""
    session_dir = Path(session_path)
    if not session_dir.exists():
        print(f"Error: Session directory {session_path} not found.")
        return

    output_dir = session_dir / "videos"
    output_dir.mkdir(exist_ok=True)

    # Iterate through all subdirectories (cameras)
    for cam_dir in session_dir.iterdir():
        if not cam_dir.is_dir() or cam_dir.name == "videos":
            continue

        print(f"🎬 Reconstructing {cam_dir.name}...")

        # Collect and sort images
        images = sorted([f for f in cam_dir.glob("*.jpg")])
        if not images:
            print(f"  No images found for {cam_dir.name}, skipping.")
            continue

        output_file = output_dir / f"{cam_dir.name}.mp4"

        try:
            with imageio.get_writer(
                output_file, format="FFMPEG", mode="I", fps=10
            ) as writer:
                for img_path in images:
                    img = imageio.imread(img_path)
                    writer.append_data(img)
            print(f"  ✅ Saved: {output_file}")
        except Exception as e:
            print(f"  ❌ Error processing {cam_dir.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run gr1_gr00t/debug_simulation.py <path_to_session_folder>")
        sys.exit(1)

    reconstruct_session(sys.argv[1])
