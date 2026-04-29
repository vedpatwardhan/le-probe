import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Standard Sparse Autoencoder (SAE) for mechanistic interpretability.
    Decomposes a dense latent space into an overcomplete sparse basis.
    """

    def __init__(self, d_model: int, d_sae: int, l1_coeff: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        # Encoder: Linear + Bias + ReLU
        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder_bias = nn.Parameter(torch.zeros(d_sae))

        # Decoder: Linear (often constrained to unit norm)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_model))

        # Initialize decoder weights to reasonable values
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        """
        x: (Batch, d_model)
        """
        # Centering
        x_centered = x - self.decoder_bias

        # Encode
        acts = F.relu(self.encoder(x_centered) + self.encoder_bias)

        # Decode
        x_reconstruct = self.decoder(acts) + self.decoder_bias

        # Loss components
        l2_loss = F.mse_loss(x_reconstruct, x)
        l1_loss = acts.abs().sum(dim=-1).mean()

        total_loss = l2_loss + self.l1_coeff * l1_loss

        return {
            "reconstruction": x_reconstruct,
            "activations": acts,
            "loss": total_loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
        }

    @torch.no_grad()
    def normalize_decoder(self):
        """Standard practice: ensure decoder columns are unit norm."""
        W = self.decoder.weight.data
        W.div_(W.norm(dim=0, keepdim=True) + 1e-8)
