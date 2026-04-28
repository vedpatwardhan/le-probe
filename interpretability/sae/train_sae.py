import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import os
import sys

# --- Path Stabilization ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from interpretability.sae.sae_model import SparseAutoencoder


def train_sae(
    activation_path,
    layer_name,
    d_sae=12288,
    lr=3e-4,
    l1_coeff=0.01,
    batch_size=4096,
    epochs=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    print(f"📂 Loading activations from {activation_path}...")
    data = torch.load(activation_path, map_location="cpu")
    activations = data[layer_name].float()

    # Flatten if necessary (B, T, D) -> (B*T, D)
    if activations.ndim == 3:
        activations = activations.reshape(-1, activations.size(-1))

    print(f"📊 Dataset size: {activations.shape}")
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize SAE
    d_model = activations.size(-1)
    sae = SparseAutoencoder(d_model=d_model, d_sae=d_sae, l1_coeff=l1_coeff).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # 3. Training Loop
    print(f"🚀 Training SAE (Expansion: {d_sae/d_model:.1f}x)...")
    for epoch in range(epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        total_l1 = 0
        total_l2 = 0

        for batch in pbar:
            x = batch[0].to(device)

            optimizer.zero_grad()
            out = sae(x)
            loss = out["loss"]

            loss.backward()
            optimizer.step()

            # Constrain decoder to unit norm
            sae.normalize_decoder()

            total_loss += loss.item()
            total_l1 += out["l1_loss"].item()
            total_l2 += out["l2_loss"].item()

            pbar.set_postfix(
                {
                    "L2": f"{out['l2_loss'].item():.4f}",
                    "L1": f"{out['l1_loss'].item():.4f}",
                }
            )

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1} complete. Avg Loss: {avg_loss:.6f}")

    # 4. Save SAE
    output_path = f"sae_{layer_name}.pt"
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": {"d_model": d_model, "d_sae": d_sae, "l1_coeff": l1_coeff},
        },
        output_path,
    )
    print(f"💾 SAE saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    # train_sae("activations.pt", "latent_bottleneck")
    print("🚂 SAE Trainer Ready.")
