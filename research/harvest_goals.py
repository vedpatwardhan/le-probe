"""
GOAL & DIAGNOSTIC HARVESTER
Role: Pre-calculates latent embeddings and diagnostic test cases.
Output: goal_gallery.pth (~5-10 MB with pixels)
"""

import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Local imports
from research.goal_mapper import GoalMapper
from research.goal_utils import get_episode_video_path, extract_frame_at_index

DIAGNOSTIC_EPISODES = [0, 30, 60, 90, 120, 149]


def harvest(model_path, dataset_root, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎬 Starting Comprehensive Harvest on {device}...")

    mapper = GoalMapper(model_path, dataset_root)
    gallery = {
        "goals": {},  # {id: goal_latent}
        "diagnostics": {},  # {id: {pixels: T,3,H,W, action: T,ADIM}}
    }

    for i in tqdm(range(500), desc="Harvesting Episodes"):
        try:
            # 1. Always Harvest the Goal Latent
            success = mapper.set_goal(episode_idx=i)
            if not success:
                print(f"\n⏹️ Reached end of dataset at episode {i}")
                break

            gallery["goals"][i] = mapper.goal_latent.cpu()

            # 2. If it's a diagnostic episode, harvest the starting context
            if i in DIAGNOSTIC_EPISODES:
                video_path = get_episode_video_path(dataset_root, i)

                # Extract frames 0, 1, 2 for the 3-frame history window
                start_frames = []
                for frame_idx in range(3):
                    frame_np = extract_frame_at_index(video_path, frame_idx)
                    # Transform to model space (3, 224, 224)
                    transformed = mapper.transform({"pixels": frame_np})["pixels"]
                    start_frames.append(transformed)

                pixels_t = torch.stack(start_frames).cpu()  # (3, 3, 224, 224)

                # For basic diagnostic, we can use zero action history or harvest actuals
                # We'll use zeros for now as per the original diagnose_mpc.py logic
                action_hist = torch.zeros(4, 64)

                gallery["diagnostics"][i] = {"pixels": pixels_t, "action": action_hist}

        except Exception as e:
            print(f"\n⚠️ Skipping episode {i} due to error: {e}")
            break

    # 3. Save the Packed Gallery
    if gallery["goals"]:
        torch.save(gallery, output_path)
        print(f"✅ Enhanced Gallery saved to: {output_path}")
        print(
            f"📈 Contains {len(gallery['goals'])} goals and {len(gallery['diagnostics'])} diagnostic test cases."
        )
    else:
        print("❌ Harvest failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="goal_gallery.pth")
    args = parser.parse_args()

    harvest(args.model, args.dataset, args.output)
