import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# --- Path Stabilization ---
# Current file is le-probe/interpretability/sae/harvest_activations.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR: le-probe/
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# We need lewm/ and lewm/le_wm/ on the path
LEWM_DIR = os.path.join(ROOT_DIR, "lewm")
LE_WM_DIR = os.path.join(LEWM_DIR, "le_wm")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if LEWM_DIR not in sys.path:
    sys.path.insert(0, LEWM_DIR)
if LE_WM_DIR not in sys.path:
    sys.path.insert(0, LE_WM_DIR)

print(f"DEBUG: sys.path: {sys.path[:5]}")
# --------------------------

from lewm.goal_mapper import GoalMapper

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


def harvest():
    # 1. Load Model
    model_path = os.path.join(ROOT_DIR, "checkpoints/v17_oracle.ckpt")
    dataset_root = os.path.join(ROOT_DIR, "datasets")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"🧠 Loading GoalMapper and Model from {model_path}...")
    gm = GoalMapper(model_path, dataset_root)
    model = gm.model
    device = gm.device

    # 2. Define Datasets
    lerobot_datasets = [
        "vedpatwardhan/gr1_pickup_cup",
        "vedpatwardhan/gr1_pickup_grasp",
    ]
    reward_pred_path = os.path.join(
        dataset_root, "vedpatwardhan/gr1_reward_pred/dataset.parquet"
    )

    all_latents = []

    # 3. Process LeRobot Datasets
    if LEROBOT_AVAILABLE:
        for repo_id in lerobot_datasets:
            print(f"📂 Harvesting LeRobot dataset: {repo_id}...")
            dataset_path = os.path.join(dataset_root, repo_id)
            try:
                # Use a smaller chunk size or limit if needed, but here we want all frames
                dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)

                batch_imgs = []
                batch_size = 32  # Reduced batch size for safety

                for i in tqdm(range(len(dataset)), desc=f"Encoding {repo_id}"):
                    frame = dataset[i]
                    img = frame["observation.images.world_center"]
                    if not isinstance(img, Image.Image):
                        if isinstance(img, torch.Tensor):
                            img = Image.fromarray(
                                img.permute(1, 2, 0).numpy().astype(np.uint8)
                            )
                        elif isinstance(img, np.ndarray):
                            if img.shape[0] == 3:
                                img = img.transpose(1, 2, 0)
                            img = Image.fromarray(img.astype(np.uint8))

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
                print(f"❌ Error loading {repo_id}: {e}")
    else:
        print("⚠️ LeRobot not available. Skipping Cup and Grasp harvests.")

    # 4. Process Reward Pred Dataset (Parquet)
    if os.path.exists(reward_pred_path):
        print(f"📂 Harvesting Reward Pred dataset: {reward_pred_path}...")
        try:
            df = pd.read_parquet(reward_pred_path)
            images = df["observation.images.world_center"].values

            batch_imgs = []
            batch_size = 32

            for i in tqdm(range(len(images)), desc="Encoding reward_pred"):
                img_array = images[i]
                if img_array.shape[0] == 3:
                    img_array = img_array.transpose(1, 2, 0)

                pil_img = Image.fromarray(img_array.astype(np.uint8))
                batch_imgs.append(gm.transform(pil_img))

                if len(batch_imgs) == batch_size or i == len(images) - 1:
                    batch_x = torch.stack(batch_imgs).to(device)
                    with torch.no_grad():
                        output = model.encoder(batch_x, interpolate_pos_encoding=True)
                        pixels_emb = output.last_hidden_state[:, 0]
                        emb = model.projector(pixels_emb)
                        all_latents.append(emb.cpu())
                    batch_imgs = []
        except Exception as e:
            print(f"❌ Error loading reward_pred: {e}")

    # 5. Save Combined Activation Dataset
    if all_latents:
        final_latents = torch.cat(all_latents, dim=0)
        output_path = os.path.join(ROOT_DIR, "interpretability/sae/activations_14k.pt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(final_latents, output_path)
        print(f"💾 Saved {len(final_latents)} total latents to {output_path}")
    else:
        print("❌ No latents captured.")


if __name__ == "__main__":
    harvest()
