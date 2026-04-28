import torch
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Path Stabilization ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from interpretability.clt.clt_model import CrossLayerTranscoder


def audit_clt(clt_path, activations_path, top_k=10):
    device = torch.device("cpu")  # Inspection is fast on CPU
    print(f"🔍 Inspecting CLT | Weights: {clt_path}")

    # 1. Load CLT and Data
    checkpoint = torch.load(clt_path, map_location="cpu")
    config = checkpoint["config"]
    norm = checkpoint["norm_stats"]

    clt = CrossLayerTranscoder(d_model=config["d_model"], d_sae=config["d_sae"])
    clt.load_state_dict(checkpoint["state_dict"])
    clt.eval()

    data = torch.load(activations_path, map_location="cpu")
    x_L = data["enc"].float()
    x_target = data["pred"].float()

    # 2. Re-apply Normalization
    x_L_norm = (x_L - norm["mean_L"]) / norm["std_L"]
    x_target_norm = (x_target - norm["mean_T"]) / norm["std_T"]

    # 3. Get CLT Activations
    with torch.no_grad():
        out = clt(x_L_norm)
        clt_acts = out["activations"]  # (N, 1024)

    # 4. Success vs Failure Split (Hypothesis: Zero-Reach fails at certain indices)
    # Since snapshots/episodes are ordered:
    # 0-2002: Snapshots
    # 2002-8414: Pickup Cup (Hard/Mixed)
    # 8414-14814: Pickup Grasp (High Success)

    success_slice = clt_acts[8414:]
    failure_slice = clt_acts[2002:8414]

    success_freq = (success_slice > 0).float().mean(dim=0)
    failure_freq = (failure_slice > 0).float().mean(dim=0)

    # Find features that fire in Success but NOT in Failure
    diff = success_freq - failure_freq
    top_success_neurons = torch.topk(diff, k=top_k)

    print(
        "\n🚀 Top 'Execution' Features (Fire in Success, Silent in Zero-Reach Failure):"
    )
    print("-" * 80)
    print(
        f"{'Feature ID':<12} | {'Success Freq':<12} | {'Failure Freq':<12} | {'Delta':<10}"
    )
    print("-" * 80)
    for i in range(top_k):
        idx = top_success_neurons.indices[i].item()
        val = top_success_neurons.values[i].item()
        print(
            f"{idx:<12} | {success_freq[idx]:<12.2%} | {failure_freq[idx]:<12.2%} | {val:<10.4f}"
        )

    # 5. Causal Contribution (Weight Inspection)
    # Which visual input directions contribute most to the top execution neuron?
    top_neuron_idx = top_success_neurons.indices[0].item()
    encoder_weights = clt.encoder.weight[top_neuron_idx]  # (192,)

    print(f"\n🧠 Feature {top_neuron_idx} Analysis:")
    print(f"This neuron likely represents the 'Reach Intent' handoff.")
    print(f"L2 Norm of Input weights: {encoder_weights.norm().item():.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clt", type=str, default="clt_weights.pt")
    parser.add_argument("--data", type=str, default="../sae/activations_dual_14k.pt")
    args = parser.parse_args()

    # Path expansion for local/relative paths
    clt_file = (
        os.path.join(SCRIPT_DIR, args.clt) if not os.path.isabs(args.clt) else args.clt
    )
    data_file = (
        os.path.join(SCRIPT_DIR, args.data)
        if not os.path.isabs(args.data)
        else args.data
    )

    audit_clt(clt_file, data_file)
