import os
import sys
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- Path Stabilization ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from interpretability.clt.clt_model import CrossLayerTranscoder


def train_clt(
    input_path,
    output_path,
    dict_size=12288,
    l1_coeff=1e-3,
    epochs=20,
    batch_size=256,
    lr=1e-3,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Training CLT | Device: {device} | Input: {input_path}")

    # 1. Load Dual-Layer Data
    if not os.path.exists(input_path):
        print(f"❌ Error: Dual activations not found at {input_path}")
        return

    data = torch.load(input_path, map_location="cpu")
    x_L = data["enc"].float()
    x_target = data["pred"].float()

    # 2. Normalization (Crucial for Cross-Layer stability)
    mean_L, std_L = x_L.mean(dim=0), x_L.std(dim=0) + 1e-6
    mean_T, std_T = x_target.mean(dim=0), x_target.std(dim=0) + 1e-6

    x_L = (x_L - mean_L) / std_L
    x_target = (x_target - mean_T) / std_T

    print(f"📊 Dataset size: {x_L.shape} -> {x_target.shape}")
    dataset = TensorDataset(x_L, x_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize CLT
    d_model = x_L.size(-1)
    model = CrossLayerTranscoder(
        d_model=d_model, d_sae=dict_size, l1_coeff=l1_coeff
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            l_in, l_target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            out = model(l_in, l_target)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            pbar.set_postfix(
                {
                    "mse": f"{out['l2_loss'].item():.4f}",
                    "sparsity": f"{(out['activations'] > 0).float().mean().item():.2%}",
                }
            )

    # 5. Save CLT with Metadata
    save_dict = {
        "state_dict": model.state_dict(),
        "config": {"d_model": d_model, "d_sae": dict_size, "l1_coeff": l1_coeff},
        "norm_stats": {
            "mean_L": mean_L,
            "std_L": std_L,
            "mean_T": mean_T,
            "std_T": std_T,
        },
    }
    torch.save(save_dict, output_path)
    print(f"💾 CLT saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../sae/activations_dual_14k.pt")
    parser.add_argument("--output", type=str, default="clt_weights.pt")
    parser.add_argument("--dict_size", type=int, default=1024)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    if not os.path.isabs(args.input):
        args.input = os.path.join(SCRIPT_DIR, args.input)
    if not os.path.isabs(args.output):
        args.output = os.path.join(SCRIPT_DIR, args.output)

    train_clt(
        args.input, args.output, args.dict_size, args.l1, args.epochs, args.batch_size
    )
