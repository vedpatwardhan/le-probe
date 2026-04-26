import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
import sys

# Add project root to sys.path for absolute imports on Colab
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from goal_mapper import GoalMapper


def train_reward_head(checkpoint_path, epochs=20, lr=1e-4):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Training on {device}")

    # 1. Load the Base Model
    print(f"🏗️  Loading Base Model: {checkpoint_path}")
    mapper = GoalMapper(model_path=checkpoint_path, dataset_root=".")
    model = mapper.model.to(device)

    # Freeze everything except the Reward Head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.reward_head.parameters():
        param.requires_grad = True

    # 2. Collect Dataset (Local first, then HF)
    snapshot_dirs = [
        "cortex-gr1/datasets/vedpatwardhan/gr1_reward_pred",  # Main consolidated repo
    ]

    # Automatic HF Fetch if missing
    if not any(os.path.exists(d) for d in snapshot_dirs):
        print(
            f"📂 Snapshots not found locally. Fetching from HF: vedpatwardhan/gr1_reward_pred..."
        )
        local_dir = "cortex-gr1/datasets/vedpatwardhan/gr1_reward_pred"
        snapshot_download(
            repo_id="vedpatwardhan/gr1_reward_pred",
            repo_type="dataset",
            local_dir=local_dir,
        )

    json_files = []
    for s_dir in snapshot_dirs:
        path = Path(s_dir)
        if path.exists():
            json_files.extend(list(path.glob("*.json")))

    if not json_files:
        print("❌ Error: No snapshots found. Check paths or HF login.")
        return

    print(f"📦 Combined Dataset: {len(json_files)} snapshots.")
    random.shuffle(json_files)

    # 90/10 Split
    split_idx = int(0.9 * len(json_files))
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    optimizer = optim.AdamW(model.reward_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_files, desc=f"Epoch {epoch+1}/{epochs}")

        for json_file in pbar:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract Image (C, H, W)
            img_list = data["observation.images.world_center"]
            img_np = np.array(img_list, dtype=np.uint8)
            if img_np.shape[-1] == 3:
                img_np = img_np.transpose(2, 0, 1)

            # Transform
            batch = mapper.transform({"pixels": img_np})
            img_input = (
                batch["pixels"].unsqueeze(0).unsqueeze(0).to(device)
            )  # (1, 1, C, H, W)

            # Ground Truth
            gt_reward = torch.tensor([[data["progress"]]], dtype=torch.float32).to(
                device
            )

            # Forward
            optimizer.zero_grad()
            outputs = model(img_input)
            pred_reward = outputs["reward"]

            loss = criterion(pred_reward, gt_reward)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for json_file in val_files:
                with open(json_file, "r") as f:
                    data = json.load(f)
                img_list = data["observation.images.world_center"]
                img_np = np.array(img_list, dtype=np.uint8)
                if img_np.shape[-1] == 3:
                    img_np = img_np.transpose(2, 0, 1)
                batch = mapper.transform({"pixels": img_np})
                img_input = batch["pixels"].unsqueeze(0).unsqueeze(0).to(device)
                gt_reward = torch.tensor([[data["progress"]]], dtype=torch.float32).to(
                    device
                )
                outputs = model(img_input)
                val_loss += criterion(outputs["reward"], gt_reward).item()

        print(
            f"✅ Epoch {epoch+1} Complete. Train Loss: {train_loss/len(train_files):.4f}, Val Loss: {val_loss/len(val_files):.4f}"
        )

    # 4. Save Final Checkpoint
    output_path = "gr1_reward_tuned_v2.ckpt"
    full_ckpt = torch.load(checkpoint_path, map_location="cpu")
    full_ckpt["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(full_ckpt, output_path)
    print(f"💾 Spectrum-Calibrated Reward Head (V2) saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="gr1-epoch=99-step=004400.ckpt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_reward_head(args.ckpt, args.epochs, args.lr)
