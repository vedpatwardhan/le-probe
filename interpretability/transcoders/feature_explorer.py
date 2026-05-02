import os
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def explore_features(weight_path, activation_dir, feature_ids, top_k=10):
    """
    Finds the top-activating examples for specific features.
    """
    # 1. Load Model & Stats
    data = torch.load(weight_path, map_location="cpu")
    sd = data["state_dict"]
    norm_stats = data["norm_stats"]
    config = data["config"]
    layer_id = config["source_layer"]

    # Extract Encoder weights: (d_model, d_sae)
    w_enc = sd["encoder.weight"].T
    b_enc = sd["encoder.bias"]

    # 2. Load Activation Metadata
    json_path = os.path.join(activation_dir, f"{layer_id}.json")
    bin_path = os.path.join(activation_dir, f"{layer_id}.bin")
    with open(json_path, "r") as f:
        meta = json.load(f)

    # Memory map the activations
    acts = np.memmap(bin_path, dtype=np.float16, mode="r", shape=tuple(meta["shape"]))

    results = {fid: [] for fid in feature_ids}

    print(f"🔍 Scanning activations for features: {feature_ids}...")

    # Batch process for speed
    batch_size = 10000
    for i in tqdm(range(0, acts.shape[0], batch_size)):
        batch = torch.from_numpy(acts[i : i + batch_size].copy()).float()

        # Normalize
        batch = (batch - norm_stats["mean"]) / norm_stats["std"]

        # Compute activations: ReLU(x @ W + b)
        # We only compute for the requested feature IDs to save memory
        feat_acts = torch.relu(batch @ w_enc[:, feature_ids] + b_enc[feature_ids])

        for idx, fid in enumerate(feature_ids):
            vals = feat_acts[:, idx]
            top_vals, top_idx = torch.topk(vals, min(top_k, vals.shape[0]))

            # Map back to global activation index
            for val, local_idx in zip(top_vals, top_idx):
                if val > 0:
                    results[fid].append((val.item(), i + local_idx.item()))

            # Keep only top_k global
            results[fid] = sorted(results[fid], key=lambda x: x[0], reverse=True)[
                :top_k
            ]

    # 3. Output Findings
    output_report = Path("feature_audit_report.json")
    with open(output_report, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✨ Audit complete. Results saved to {output_report}")
    for fid in feature_ids:
        print(
            f"Feature {fid}: Max Activation = {results[fid][0][0] if results[fid] else 0:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to .pt weight file"
    )
    parser.add_argument(
        "--acts", type=str, required=True, help="Path to activations directory"
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=[0, 10, 100, 1000],
        help="Feature IDs to audit",
    )
    args = parser.parse_args()

    explore_features(args.weights, args.acts, args.features)
