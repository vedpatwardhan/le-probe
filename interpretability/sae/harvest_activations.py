import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# --- Path Stabilization (Handles local and Colab environments) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LEWM_DIR = os.path.join(ROOT_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

for path in [ROOT_DIR, LEWM_DIR, LE_WM_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from lewm.goal_mapper import GoalMapper

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# -----------------------------------------------------------------------------


def harvest(ckpt_path, dataset_root, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Harvest | Device: {device} | CKPT: {ckpt_path}")

    # 1. Load Model
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Model not found at {ckpt_path}")
        return

    gm = GoalMapper(ckpt_path, dataset_root)
    model = gm.model.to(device)
    model.eval()

    all_latents = []

    # 2. Dataset Definitions
    lerobot_datasets = [
        "vedpatwardhan/gr1_pickup_cup",
        "vedpatwardhan/gr1_pickup_grasp",
    ]
    reward_pred_path = os.path.join(
        dataset_root, "vedpatwardhan/gr1_reward_pred/dataset.parquet"
    )

    # 3. Harvest LeRobot Datasets
    if LEROBOT_AVAILABLE:
        for repo_id in lerobot_datasets:
            ds_path = os.path.join(dataset_root, repo_id)
            if not os.path.exists(ds_path):
                print(f"⚠️ Dataset path missing: {ds_path}. Skipping.")
                continue

            print(f"📂 Harvesting LeRobot: {repo_id}")
            try:
                dataset = LeRobotDataset(repo_id=repo_id, root=ds_path)
                batch_imgs = []
                batch_size = 64

                for i in tqdm(range(len(dataset)), desc=f"Processing {repo_id}"):
                    frame = dataset[i]
                    img = frame["observation.images.world_center"]
                    batch_imgs.append(gm.transform(img))

                    if len(batch_imgs) == batch_size or i == len(dataset) - 1:
                        batch_x = torch.stack(batch_imgs).to(device)
                        with torch.no_grad():
                            output = model.encoder(
                                batch_x, interpolate_pos_encoding=True
                            )
                            pixels_emb = output.last_hidden_state[:, 0]
                            emb = model.projector(pixels_emb)
                            all_latents.append(emb.cpu())
                        batch_imgs = []
            except Exception as e:
                print(f"❌ Error harvesting {repo_id}: {e}")
    else:
        print("⚠️ LeRobot not installed. Skipping Cup/Grasp datasets.")

    # 4. Harvest Snapshot Parquet
    if os.path.exists(reward_pred_path):
        print(f"📂 Harvesting Snapshots: {reward_pred_path}")
        try:
            df = pd.read_parquet(reward_pred_path)
            images = df["observation.images.world_center"].values
            batch_imgs = []
            batch_size = 64

            for i in tqdm(range(len(images)), desc="Processing snapshots"):
                img_array = images[i]
                if img_array.shape[0] == 3:
                    img_array = img_array.transpose(1, 2, 0)
                batch_imgs.append(
                    gm.transform(Image.fromarray(img_array.astype(np.uint8)))
                )

                if len(batch_imgs) == batch_size or i == len(images) - 1:
                    batch_x = torch.stack(batch_imgs).to(device)
                    with torch.no_grad():
                        output = model.encoder(batch_x, interpolate_pos_encoding=True)
                        pixels_emb = output.last_hidden_state[:, 0]
                        emb = model.projector(pixels_emb)
                        all_latents.append(emb.cpu())
                    batch_imgs = []
        except Exception as e:
            print(f"❌ Error harvesting snapshots: {e}")

    # 5. Save Final Activation Set
    if all_latents:
        final_latents = torch.cat(all_latents, dim=0)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(final_latents, output_path)
        print(f"💾 Success! Saved {len(final_latents)} latents to {output_path}")
    else:
        print("❌ No latents were harvested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(ROOT_DIR, "..", "gr1_reward_tuned_v2.ckpt"),
    )
    parser.add_argument("--root", type=str, default=os.path.join(ROOT_DIR, "datasets"))
    parser.add_argument(
        "--out", type=str, default=os.path.join(SCRIPT_DIR, "activations_14k.pt")
    )
    args = parser.parse_args()

    harvest(args.ckpt, args.root, args.out)
