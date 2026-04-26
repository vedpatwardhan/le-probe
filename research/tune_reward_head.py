import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Project Imports
import sys
from huggingface_hub import snapshot_download

sys.path.append("cortex-gr1")
sys.path.append("cortex-gr1/research")
sys.path.append("cortex-gr1/le_wm")

from goal_mapper import GoalMapper


class SnapshotDataset(Dataset):
    def __init__(self, snapshots_dir, transform):
        self.snapshots_dir = Path(snapshots_dir)
        self.json_files = sorted([f for f in self.snapshots_dir.glob("*.json")])
        self.transform = transform
        print(f"📦 Dataset initialized with {len(self.json_files)} snapshots.")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        with open(self.json_files[idx], "r") as f:
            data = json.load(f)

        # Extract and Transform Image (C, H, W) -> (H, W, C) for transform -> (C, H, W) normalized
        img_list = data["observation.images.world_center"]
        img_np = np.array(img_list, dtype=np.uint8).transpose(1, 2, 0)

        batch = self.transform({"pixels": img_np})
        pixel_tensor = batch["pixels"]  # (C, H, W) normalized

        reward = float(data.get("progress", 0.0))

        return pixel_tensor, torch.tensor([reward], dtype=torch.float32)


def train_reward_head(
    checkpoint_path, snapshots_dir, epochs=20, lr=1e-4, batch_size=32
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Training on device: {device}")

    # 1. Load Model
    print(f"🏗️  Loading Model from {checkpoint_path}...")
    mapper = GoalMapper(model_path=checkpoint_path, dataset_root=".")
    model = mapper.model.to(device)

    # 2. Freeze everything except reward_head
    for param in model.parameters():
        param.requires_grad = False

    for param in model.reward_head.parameters():
        param.requires_grad = True

    print("❄️  JEPA Backbone Frozen. 🔥 Reward Head ACTIVE.")

    # 3. Setup Data
    full_dataset = SnapshotDataset(snapshots_dir, mapper.transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Split: {train_size} training samples, {val_size} validation samples.")

    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.reward_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. Training Loop
    print(f"🏋️  Starting Fine-Tuning for {epochs} epochs...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.reward_head.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for pixels, targets in pbar:
            pixels, targets = pixels.to(device), targets.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                img_input = pixels.unsqueeze(1)
                info = model.encode({"pixels": img_input})
                z = info["emb"]

            pred_reward = model.reward_head(z).squeeze(1)
            loss = criterion(pred_reward, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.reward_head.eval()
        val_loss = 0
        with torch.no_grad():
            for pixels, targets in val_loader:
                pixels, targets = pixels.to(device), targets.to(device)
                img_input = pixels.unsqueeze(1)
                info = model.encode({"pixels": img_input})
                z = info["emb"]

                pred_reward = model.reward_head(z).squeeze(1)
                val_loss += criterion(pred_reward, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"📈 Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

    # 6. Save Updated Weights
    # We save as a new checkpoint to avoid messing with the original
    output_path = "gr1_reward_tuned.ckpt"

    # Preserve the original structure but update the state_dict
    full_ckpt = torch.load(checkpoint_path, map_location="cpu")
    full_ckpt["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(full_ckpt, output_path)
    print(f"✅ Fine-tuning complete! Saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="gr1-epoch=99-step=004400.ckpt")
    parser.add_argument(
        "--snapshots",
        type=str,
        default="cortex-gr1/datasets/vedpatwardhan/gr1_reward_pred",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Automatic HF Fetch Logic
    if not os.path.exists(args.snapshots):
        print(f"📂 Local snapshots not found at {args.snapshots}")
        print(f"📥 Fetching from Hugging Face: vedpatwardhan/gr1_reward_pred...")
        try:
            # Download to the expected local path
            snapshot_download(
                repo_id="vedpatwardhan/gr1_reward_pred",
                repo_type="dataset",
                local_dir=args.snapshots,
            )
            print(f"✅ Download complete. Files saved to {args.snapshots}")
        except Exception as e:
            print(f"❌ Error downloading from HF: {e}")
            print("💡 Please ensure you are logged in with 'huggingface-cli login'")
            sys.exit(1)

    train_reward_head(args.ckpt, args.snapshots, args.epochs, args.lr, args.batch_size)
