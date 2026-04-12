"""
GOAL PERCEPTION UTILS
Role: Dataset management and success-state visual extraction.

This script handles the 'Visual Memory' of the agent. It:
1. Locates the success frame (last frame) for any given episode in the dataset.
2. Decodes the video using torchcodec (high-performance AV1 support).
3. Prepares the goal image for latent encoding by the JEPA model.
"""

from pathlib import Path

# Try to import torchcodec for high-performance AV1 decoding
try:
    import torchcodec

    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False


def get_goal_pixels(video_path):
    """
    Extracts the last frame (success state) from an episode video.
    Uses torchcodec for robust AV1 support on Colab/Linux.
    """
    if not HAS_TORCHCODEC:
        raise ImportError(
            "torchcodec is required for MPC goal extraction. "
            "Please install it via 'pip install torchcodec'."
        )

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return None

    try:
        # 1. Initialize Decoder
        decoder = torchcodec.decoders.VideoDecoder(str(video_path))

        # 2. Target the last frame (Success State)
        num_frames = decoder.metadata.num_frames
        last_frame_idx = num_frames - 1

        # 3. Decode specific frame
        # Torchcodec returns values in [0, 1] as float32 tensors (C, H, W)
        frame_batch = decoder.get_frames_at(indices=[last_frame_idx])
        goal_pixels = frame_batch.data[0]  # Take first (and only) frame

        print(f"🎯 Goal established: Frame {last_frame_idx} of {video_path.name}")
        return goal_pixels

    except Exception as e:
        print(f"❌ torchcodec failed to read video: {e}")
        return None


def get_episode_video_path(
    dataset_root, episode_idx, camera_key="observation.images.world_center"
):
    """
    Finds the MP4 file for a given episode index.
    Matches the LeRobot/HF directory structure.
    """
    root = Path(dataset_root)
    video_path = (
        root / "videos" / camera_key / "chunk-000" / f"file-{episode_idx:03d}.mp4"
    )
    return video_path
