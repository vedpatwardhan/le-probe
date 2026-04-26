import sys

sys.path.append("cortex-gr1")
sys.path.append("cortex-gr1/research")
sys.path.append("cortex-gr1/le_wm")
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Project-specific imports
from goal_mapper import GoalMapper


def predict_reward(snapshot, checkpoint_path, goal_gallery_path):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    # 1. Load Model
    print(f"🏗️  Loading Model from {checkpoint_path}...")
    mapper = GoalMapper(model_path=checkpoint_path, dataset_root=".")
    mapper.model.to(device)
    mapper.model.eval()

    # 2. Load Goal Gallery
    print(f"🎯 Loading Goal Gallery from {goal_gallery_path}...")
    gallery_data = torch.load(goal_gallery_path, map_location=device)
    goal_embs = gallery_data["goals"]

    if isinstance(goal_embs, dict):
        # The gallery stores goals as a dict of {idx: tensor}
        # We stack them into a single tensor for batch processing
        goal_embs = torch.stack([v.to(device) for v in goal_embs.values()])

    # Goal Gallery tensors are often (N, 1, T, D) or (N, T, D)
    # We reduce them to (N, D) by taking the final frame of each sequence
    while goal_embs.ndim > 2:
        goal_embs = goal_embs[:, -1]

    # 3. Load Snapshot JSON
    with open(snapshot, "r") as f:
        data = json.load(f)

    # Extract Image (C, H, W)
    img_list = data["observation.images.world_center"]
    img_np = np.array(img_list, dtype=np.uint8).transpose(
        1, 2, 0
    )  # (C, H, W) -> (H, W, C) for transform
    print(f"DEBUG: img_np shape: {img_np.shape}")

    # Use OFFICIAL transform pipeline (Strict Parity)
    batch = mapper.transform({"pixels": img_np})
    img_input = batch["pixels"].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)

    # 4. Inference
    with torch.no_grad():
        # Encode
        info = mapper.model.encode({"pixels": img_input})
        z = info["emb"]  # (1, 1, 192)

        # Pred Reward Head
        pred_reward = mapper.model.reward_head(z).item()

        # Distance to Goal Gallery (Euclidean Distance)
        # We find the distance to the closest goal in the gallery, matching lewm_server.py logic
        # z: (1, 1, 192) -> (1, 192), goal_embs: (N, 192)
        flat_z = z.view(-1, z.size(-1))
        dists = torch.cdist(flat_z, goal_embs)  # (1, N)
        min_dist = dists.min().item()

    # 5. Report
    print("\n" + "=" * 40)
    print(f"🔍 ANALYSIS: {Path(snapshot).name}")
    print("=" * 40)
    print(f"✅ Ground Truth Reward:  {data.get('progress', 'N/A'):.4f}")
    print(f"🤖 Predicted Reward:     {pred_reward:.4f}")
    print(f"🎯 Best Goal Distance:   {min_dist:.4f}")
    print("=" * 40)

    if pred_reward > 8.0 and data.get("progress", 0) < 1.0:
        print(
            "🚨 WARNING: High Reward Prediction for Low Progress State! (Reward Sink Detected)"
        )
    elif min_dist < 0.01:
        print("✨ EXCELLENT: This state is very close to a known goal latent.")
    else:
        print("💡 INFO: State is physically grounded but not yet at goal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, help="Path to snapshot JSON")
    parser.add_argument("--ckpt", type=str, default="gr1-epoch=99-step=004400.ckpt")
    parser.add_argument("--gallery", type=str, default="goal_gallery.pth")
    args = parser.parse_args()

    predict_reward(args.snapshot, args.ckpt, args.gallery)
