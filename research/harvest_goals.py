"""
GOAL HARVESTER
Role: Pre-calculates latent embeddings for all task success states.
Purpose: Decouples the Inference Server from the 100GB dataset.
Output: goal_gallery.pth (~115 KB)
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Local imports from the research layer
from research.goal_mapper import GoalMapper


def harvest(model_path, dataset_root, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎬 Starting Goal Harvest on {device}...")

    # 1. Initialize the Mapper (Needs the dataset for harvesting)
    mapper = GoalMapper(model_path, dataset_root)

    gallery = {}

    # 2. Iterate through all episodes (Assuming 0-149 for gr1_pickup)
    # The script will stop automatically if it runs out of episodes
    for i in tqdm(range(500), desc="Harvesting Success Latents"):
        try:
            success = mapper.set_goal(episode_idx=i)
            if success:
                # Store the encoded latent: goal_latent is (1, 1, 192)
                gallery[i] = mapper.goal_latent.cpu()
            else:
                print(f"\n⏹️ Reached end of dataset at episode {i}")
                break
        except Exception as e:
            print(f"\n⚠️ Skipping episode {i} due to error: {e}")
            break

    # 3. Save the Gallery
    if gallery:
        torch.save(gallery, output_path)
        print(f"✅ Goal Gallery saved to: {output_path} ({len(gallery)} entries)")
        print(f"🚀 You can now run lewm_server.py without the dataset!")
    else:
        print("❌ No goals were harvested. Check your dataset path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to gr1-epoch=99.ckpt"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to local dataset snapshots"
    )
    parser.add_argument("--output", type=str, default="goal_gallery.pth")
    args = parser.parse_args()

    harvest(args.model, args.dataset, args.output)
