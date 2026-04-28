import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download

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


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_batch(model, device, batch_imgs):
    """Unified encoding logic for all datasets."""
    batch_x = torch.stack(batch_imgs).to(device)
    output = model.encoder(batch_x, interpolate_pos_encoding=True)
    pixels_emb = output.last_hidden_state[:, 0]  # CLS token
    emb = model.projector(pixels_emb)
    return emb.cpu()


def harvest(ckpt_path, dataset_root, output_path):
    device = get_best_device()
    print(f"🚀 Initializing Harvest | Device: {device} | CKPT: {ckpt_path}")

    # 1. Load Model
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Model not found at {ckpt_path}")
        return

    gm = GoalMapper(ckpt_path, dataset_root)
    model = gm.model.to(device)
    model.eval()

    all_latents = []
    batch_size = 64

    # 2. Dataset Definitions
    lerobot_datasets = [
        "vedpatwardhan/gr1_pickup_cup",
        "vedpatwardhan/gr1_pickup_grasp",
    ]
    reward_pred_path = os.path.join(
        dataset_root, "vedpatwardhan/gr1_reward_pred/dataset.parquet"
    )

    # 3. Harvest Snapshot Parquet (FAIL FAST)
    if not os.path.exists(reward_pred_path):
        print(f"📡 Snapshot parquet not found. Attempting Hub fetch...")
        try:
            reward_pred_path = hf_hub_download(
                repo_id="vedpatwardhan/gr1_reward_pred",
                filename="dataset.parquet",
                repo_type="dataset",
                local_dir=os.path.dirname(reward_pred_path),
            )
        except Exception as e:
            print(f"⚠️ Could not fetch snapshots from Hub: {e}")

    if os.path.exists(reward_pred_path):
        print(f"📂 Harvesting Snapshots: {reward_pred_path}")
        try:
            df = pd.read_parquet(reward_pred_path)
            images = df["observation.images.world_center"].values
            batch = []

            for i, raw_img in enumerate(tqdm(images, desc="Processing snapshots")):
                try:
                    # Reconstruction logic from tune_reward_head.py
                    img = np.stack(
                        [
                            np.array(raw_img[0].tolist(), dtype=np.uint8),
                            np.array(raw_img[1].tolist(), dtype=np.uint8),
                            np.array(raw_img[2].tolist(), dtype=np.uint8),
                        ],
                        axis=-1,
                    )
                except Exception:
                    img = np.array(raw_img).astype(np.uint8)
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)

                batch.append(gm.transform({"pixels": img})["pixels"])

                if len(batch) == batch_size or i == len(images) - 1:
                    all_latents.append(encode_batch(model, device, batch))
                    batch = []
        except Exception as e:
            print(f"❌ Error harvesting snapshots: {e}")
            traceback.print_exc()

    # 4. Harvest LeRobot Datasets
    if LEROBOT_AVAILABLE:
        for repo_id in lerobot_datasets:
            ds_path = os.path.join(dataset_root, repo_id)
            print(f"📂 Harvesting LeRobot: {repo_id}")
            try:
                dataset = LeRobotDataset(repo_id=repo_id, root=ds_path)
                batch = []

                for i in tqdm(range(len(dataset)), desc=f"Processing {repo_id}"):
                    img = dataset[i]["observation.images.world_center"]
                    if not (torch.is_tensor(img) or isinstance(img, np.ndarray)):
                        img = np.array(img)

                    batch.append(gm.transform({"pixels": img})["pixels"])

                    if len(batch) == batch_size or i == len(dataset) - 1:
                        all_latents.append(encode_batch(model, device, batch))
                        batch = []
            except Exception as e:
                print(f"❌ Error harvesting {repo_id}: {e}")
                traceback.print_exc()

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
