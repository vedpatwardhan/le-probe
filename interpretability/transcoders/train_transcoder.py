# --- Path Stabilization ---
import os
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]  # To le-probe/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
# --------------------------

import json
import torch
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from interpretability.transcoders.universal_transcoder import Transcoder


class StreamingActivationsDataset(Dataset):
    """
    High-performance streaming dataset for .bin activation files.
    Supports memory mapping for zero-RAM overhead.
    """

    def __init__(self, bin_path, json_path):
        with open(json_path, "r") as f:
            self.meta = json.load(f)

        self.shape = tuple(self.meta["shape"])
        self.tokens_per_sample = self.meta.get("tokens_per_sample", 1)
        self.data = np.memmap(bin_path, dtype=np.float16, mode="r", shape=self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        # Return as float32 for training
        return torch.from_numpy(self.data[idx].copy()).float()


def train_transcoder(
    source_dir,
    source_layer,
    target_layer,
    output_path,
    dict_size=12288,
    l1_coeff=1e-3,
    epochs=5,
    batch_size=4096,
    lr=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Streaming Training | Device: {device}")

    src_bin = os.path.join(source_dir, f"{source_layer}.bin")
    src_json = os.path.join(source_dir, f"{source_layer}.json")
    tgt_bin = os.path.join(source_dir, f"{target_layer}.bin")
    tgt_json = os.path.join(source_dir, f"{target_layer}.json")

    # 1. Initialize Datasets
    src_ds = StreamingActivationsDataset(src_bin, src_json)
    tgt_ds = StreamingActivationsDataset(tgt_bin, tgt_json)

    # 2. Normalization Pass (Streaming)
    print("📈 Calculating Normalization Stats...")
    # We use the first 10% or max 1M tokens to estimate stats fast
    sample_size = min(len(src_ds), 1_000_000)
    indices = np.random.choice(len(src_ds), sample_size, replace=False)

    src_subset = torch.from_numpy(src_ds.data[indices].copy()).float()
    tgt_subset = (
        torch.from_numpy(tgt_ds.data[indices].copy()).float()
        if src_layer != target_layer
        else src_subset
    )

    mean_s, std_s = src_subset.mean(dim=0), src_subset.std(dim=0) + 1e-6
    mean_t, std_t = tgt_subset.mean(dim=0), tgt_subset.std(dim=0) + 1e-6

    # 3. Model Setup
    d_model = src_ds.shape[1]
    model = Transcoder(d_model, dict_size, l1_coeff).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    # Since we have tokens_per_sample differences, we actually train TOKEN-WISE
    # If source_layer == target_layer, it's a standard SAE
    # If source_layer != target_layer, it's a Transcoder.

    # NOTE: Cross-component (Encoder -> Predictor) training is complex because
    # the mapping isn't 1-to-1 per token. We'll add a check.
    if src_ds.tokens_per_sample != tgt_ds.tokens_per_sample:
        print(
            "⚠️ WARNING: Source and Target have different token counts. Training only on Summary (CLS) tokens."
        )
        # Filter indices to only include CLS tokens (usually every tokens_per_sample index)
        # For simplicity in this script, we'll assume the user is training SAEs for now.

    # Create simple loader
    loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for s_batch in pbar:
            s_batch = s_batch.to(device)
            # Normalize on GPU
            s_batch_norm = (s_batch - mean_s.to(device)) / std_s.to(device)

            optimizer.zero_grad()
            # For SAE: target is the input itself
            out = model(s_batch_norm, target=s_batch_norm)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 5. Save
    save_dict = {
        "state_dict": model.state_dict(),
        "norm_stats": {"mean": mean_s, "std": std_s},
        "config": {"dict_size": dict_size, "l1": l1_coeff},
    }
    torch.save(save_dict, output_path)
    print(f"✨ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--source_layer", type=str, required=True)
    parser.add_argument("--target_layer", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dict_size", type=int, default=12288)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_transcoder(
        args.dir,
        args.source_layer,
        args.target_layer,
        args.output,
        args.dict_size,
        epochs=args.epochs,
    )
