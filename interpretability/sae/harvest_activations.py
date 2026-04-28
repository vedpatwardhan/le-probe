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
def encode_dual_layer(model, device, batch_imgs):
    """
    Harvests activations from both the Encoder and the Predictor for CLT training.
    """
    batch_x = torch.stack(batch_imgs).to(device)

    # 1. Encoder Path
    enc_out = model.encoder(batch_x, interpolate_pos_encoding=True)
    pixels_emb = enc_out.last_hidden_state[:, 0]
    enc_latents = model.projector(pixels_emb)  # (B, 192)

    # 2. Predictor Path (Handoff)
    # We pass the enc_latents into the predictor with zero actions to see the "base" predictive state
    # Note: GR1Embedder expects 32d input (Action 14d + State 18d)
    dummy_act = torch.zeros(batch_x.size(0), 1, 32).to(device)
    act_emb = model.action_encoder(dummy_act)  # Encode 32d -> 192d
    pred_latents = model.predict(enc_latents.unsqueeze(1), act_emb).squeeze(1)

    return enc_latents.cpu(), pred_latents.cpu()


def harvest(ckpt_path, dataset_root, output_path):
    device = get_best_device()
    print(f"🚀 Dual-Layer Harvest (CLT) | Device: {device} | CKPT: {ckpt_path}")

    gm = GoalMapper(ckpt_path, dataset_root)
    model = gm.model.to(device)
    model.eval()

    all_enc = []
    all_pred = []
    batch_size = 64

    # Dataset Definitions
    lerobot_datasets = [
        "vedpatwardhan/gr1_pickup_cup",
        "vedpatwardhan/gr1_pickup_grasp",
    ]
    reward_pred_path = os.path.join(
        dataset_root, "vedpatwardhan/gr1_reward_pred/dataset.parquet"
    )

    # [Snapshot Harvest Code Simplified for Dual Layer]
    if os.path.exists(reward_pred_path):
        print(f"📂 Harvesting Snapshots: {reward_pred_path}")
        df = pd.read_parquet(reward_pred_path)
        images = df["observation.images.world_center"].values
        batch = []
        for i, raw_img in enumerate(tqdm(images, desc="Snapshots")):
            try:
                img = np.stack(
                    [np.array(raw_img[c].tolist(), dtype=np.uint8) for c in range(3)],
                    axis=-1,
                )
            except:
                img = np.array(raw_img).astype(np.uint8)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)

            batch.append(gm.transform({"pixels": img})["pixels"])
            if len(batch) == batch_size or i == len(images) - 1:
                e, p = encode_dual_layer(model, device, batch)
                all_enc.append(e)
                all_pred.append(p)
                batch = []

    # [LeRobot Harvest Code Simplified for Dual Layer]
    if LEROBOT_AVAILABLE:
        for repo_id in lerobot_datasets:
            print(f"📂 Harvesting LeRobot: {repo_id}")
            dataset = LeRobotDataset(
                repo_id=repo_id, root=os.path.join(dataset_root, repo_id)
            )
            batch = []
            for i in tqdm(range(len(dataset)), desc=repo_id):
                img = dataset[i]["observation.images.world_center"]
                if not (torch.is_tensor(img) or isinstance(img, np.ndarray)):
                    img = np.array(img)
                batch.append(gm.transform({"pixels": img})["pixels"])
                if len(batch) == batch_size or i == len(dataset) - 1:
                    e, p = encode_dual_layer(model, device, batch)
                    all_enc.append(e)
                    all_pred.append(p)
                    batch = []

    if all_enc:
        torch.save(
            {"enc": torch.cat(all_enc, dim=0), "pred": torch.cat(all_pred, dim=0)},
            output_path,
        )
        print(f"💾 Dual-Layer Latents saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(ROOT_DIR, "..", "gr1_reward_tuned_v2.ckpt"),
    )
    parser.add_argument("--root", type=str, default=os.path.join(ROOT_DIR, "datasets"))
    parser.add_argument(
        "--out", type=str, default=os.path.join(SCRIPT_DIR, "activations_dual_14k.pt")
    )
    args = parser.parse_args()
    harvest(args.ckpt, args.root, args.out)
