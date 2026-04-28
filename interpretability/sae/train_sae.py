import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size

        # Encoder: x -> f
        self.encoder = nn.Linear(input_dim, dict_size)
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))

        # Decoder: f -> x_hat
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        # Weight Initialization
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        # f = ReLU(W_enc(x - b_dec) + b_enc)
        x_cent = x - self.decoder_bias
        latent = torch.relu(self.encoder(x_cent) + self.encoder_bias)

        # x_hat = W_dec(f) + b_dec
        x_hat = self.decoder(latent) + self.decoder_bias
        return x_hat, latent


def train_sae(
    input_path,
    output_path,
    dict_size=1024,
    l1_coeff=1e-3,
    epochs=50,
    batch_size=256,
    lr=1e-3,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"🚀 Training SAE | Device: {device} | Input: {input_path}")

    # 1. Load Latents
    if not os.path.exists(input_path):
        print(f"❌ Error: Latents not found at {input_path}")
        return

    all_latents = torch.load(input_path, map_location="cpu")
    if isinstance(all_latents, list):
        all_latents = torch.cat(all_latents, dim=0)

    # Normalize latents (Unit Variance)
    mean = all_latents.mean(dim=0)
    std = all_latents.std(dim=0) + 1e-6
    all_latents = (all_latents - mean) / std

    dataset = TensorDataset(all_latents)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model
    input_dim = all_latents.shape[1]
    model = SparseAutoencoder(input_dim, dict_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_l1 = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for (x,) in pbar:
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, f = model(x)

            # Loss = MSE(x, x_hat) + lambda * L1(f)
            mse_loss = nn.MSELoss()(x_hat, x)
            l1_loss = f.abs().sum(dim=-1).mean()
            loss = mse_loss + l1_coeff * l1_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()

            pbar.set_postfix(
                {
                    "mse": f"{mse_loss.item():.4f}",
                    "l1": f"{l1_loss.item():.2f}",
                    "sparsity": f"{(f > 0).float().mean().item():.2%}",
                }
            )

    # 4. Save Model
    save_dict = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "dict_size": dict_size,
        "l1_coeff": l1_coeff,
        "norm_stats": {"mean": mean, "std": std},
    }
    torch.save(save_dict, output_path)
    print(f"💾 SAE Training Complete! Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="activations_14k.pt",
        help="Path to harvested latents",
    )
    parser.add_argument(
        "--output", type=str, default="sae_weights.pt", help="Path to save SAE weights"
    )
    parser.add_argument(
        "--dict_size", type=int, default=1024, help="Number of dictionary features"
    )
    parser.add_argument(
        "--l1", type=float, default=1e-3, help="L1 Sparsity coefficient"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    # Ensure input path is absolute or relative to script
    if not os.path.isabs(args.input):
        args.input = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.input
        )
    if not os.path.isabs(args.output):
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.output
        )

    train_sae(
        args.input,
        args.output,
        args.dict_size,
        args.l1,
        args.epochs,
        args.batch_size,
        args.lr,
    )
