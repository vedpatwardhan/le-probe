import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

# We'll use the same class definition as in train_sae.py
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        x_cent = x - self.decoder_bias
        latent = torch.relu(self.encoder(x_cent) + self.encoder_bias)
        return latent


def inspect(latents_path, sae_path, top_k=10):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # 1. Load Data
    print(f"📂 Loading SAE and Latents...")
    sae_data = torch.load(sae_path, map_location="cpu")
    latents = torch.load(latents_path, map_location="cpu")

    # Initialize Model
    model = SparseAutoencoder(sae_data["input_dim"], sae_data["dict_size"])
    model.load_state_dict(sae_data["state_dict"])
    model.to(device).eval()

    # 2. Normalize and Encode
    mean = sae_data["norm_stats"]["mean"]
    std = sae_data["norm_stats"]["std"]
    norm_latents = (latents - mean) / std

    print(
        f"🧠 Encoding {len(norm_latents)} latents into {sae_data['dict_size']} features..."
    )
    with torch.no_grad():
        activations = model(norm_latents.to(device)).cpu()

    # 3. Contrastive Audit (Success vs. Horrible)
    # Based on our harvest order:
    # [0:1000] = Horrible, [1000:1600] = Mediocre, [1600:2002] = Success
    horrible_act = activations[0:1000]
    success_act = activations[1600:2000]

    mean_horrible = horrible_act.mean(dim=0)
    mean_success = success_act.mean(dim=0)

    # Scoring: Features that are high in success but low in horrible
    scores = mean_success - mean_horrible
    top_indices = torch.argsort(scores, descending=True)[:top_k]

    print("\n" + "=" * 50)
    print("🏆 TOP FEATURES: SUCCESS-CORRELATED (Potential 'Reach/Contact' Neurons)")
    print("=" * 50)
    for i, idx in enumerate(top_indices):
        idx = idx.item()
        s_val = mean_success[idx].item()
        h_val = mean_horrible[idx].item()
        print(
            f"#{i+1} | Feature {idx:4d} | Success: {s_val:6.2f} | Horrible: {h_val:6.2f} | Diff: {scores[idx]:6.2f}"
        )

    # 4. Global Sparsity Audit
    feature_freq = (activations > 0).float().mean(dim=0)
    dead_features = (feature_freq == 0).sum().item()
    print("\n" + "=" * 50)
    print("📊 GLOBAL DICTIONARY STATS")
    print("=" * 50)
    print(f"Total Features: {sae_data['dict_size']}")
    print(
        f"Dead Features:  {dead_features} ({(dead_features/sae_data['dict_size']):.1%})"
    )
    print(f"Avg Frequency:  {feature_freq.mean().item():.2%}")

    # Top Frequency Features
    high_freq_idx = torch.argsort(feature_freq, descending=True)[:5]
    print("\nMost Frequent Features (Potential 'Background/Static' features):")
    for idx in high_freq_idx:
        print(f"Feature {idx.item():4d} | Frequency: {feature_freq[idx.item()]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=str, default="activations_14k.pt")
    parser.add_argument("--sae", type=str, default="sae_weights.pt")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # Handle paths relative to script
    if not os.path.isabs(args.latents):
        args.latents = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.latents
        )
    if not os.path.isabs(args.sae):
        args.sae = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.sae)

    inspect(args.latents, args.sae, args.top_k)
