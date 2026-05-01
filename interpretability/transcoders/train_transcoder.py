import os
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from interpretability.transcoders.universal_transcoder import Transcoder


def train_transcoder(
    source_path,
    target_path,
    output_path,
    dict_size=12288,
    l1_coeff=1e-3,
    epochs=30,
    batch_size=256,
    lr=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Transcoder | Device: {device}")
    print(f"  - Source: {source_path}")
    print(f"  - Target: {target_path}")

    # 1. Load Data
    x_source = torch.load(source_path, map_location="cpu").float()
    x_target = (
        torch.load(target_path, map_location="cpu").float()
        if source_path != target_path
        else x_source
    )

    # 2. Normalize (Unit Variance for both Source and Target)
    mean_s, std_s = x_source.mean(dim=0), x_source.std(dim=0) + 1e-6
    mean_t, std_t = x_target.mean(dim=0), x_target.std(dim=0) + 1e-6

    x_s_norm = (x_source - mean_s) / std_s
    x_t_norm = (x_target - mean_t) / std_t

    dataset = TensorDataset(x_s_norm, x_t_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Model
    d_model = x_source.shape[1]
    model = Transcoder(d_model, dict_size, l1_coeff).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss, total_mse, total_l1 = 0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for s_batch, t_batch in pbar:
            s_batch, t_batch = s_batch.to(device), t_batch.to(device)
            optimizer.zero_grad()

            out = model(s_batch, target=t_batch)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            model.normalize_decoder()

            total_loss += loss.item()
            total_mse += out["l2_loss"].item()
            total_l1 += out["l1_loss"].item()

            pbar.set_postfix(
                {
                    "mse": f"{out['l2_loss'].item():.4f}",
                    "sparsity": f"{(out['activations'] > 0).float().mean().item():.2%}",
                }
            )

    # 5. Save with Metadata
    save_dict = {
        "state_dict": model.state_dict(),
        "input_dim": d_model,
        "dict_size": dict_size,
        "l1_coeff": l1_coeff,
        "norm_stats": {
            "source_mean": mean_s,
            "source_std": std_s,
            "target_mean": mean_t,
            "target_std": std_t,
        },
    }
    torch.save(save_dict, output_path)
    print(f"💾 Transcoder saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, required=True, help="Path to source activations"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target activations (same as source for SAE)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output weights path")
    parser.add_argument("--dict_size", type=int, default=12288)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    train_transcoder(
        args.source,
        args.target,
        args.output,
        args.dict_size,
        args.l1,
        args.epochs,
        args.batch_size,
        args.lr,
    )
