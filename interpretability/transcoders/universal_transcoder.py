import torch
import torch.nn as nn
import torch.nn.functional as F


class Transcoder(nn.Module):
    """
    Universal Transcoder Architecture.
    Can operate as an SAE (Identity: Layer L -> Layer L)
    or a CLT (Transition: Layer L -> Layer L+1).
    """

    def __init__(self, d_model: int, d_dict: int, l1_coeff: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        self.l1_coeff = l1_coeff

        # Encoder: Linear + Bias + ReLU
        # Probes the input space (Source Layer)
        self.encoder = nn.Linear(d_model, d_dict)
        self.b_enc = nn.Parameter(torch.zeros(d_dict))

        # Decoder: Linear
        # Projects back into the output space (Target Layer)
        self.decoder = nn.Linear(d_dict, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Orthogonal initialization for stability
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x, target=None):
        """
        x: Input activations (Batch, d_model)
        target: Target activations for training (Optional)
        """
        # Centering (Standard SAE/CLT practice)
        x_centered = x - self.b_dec

        # Encode: Map to sparse latent space
        acts = F.relu(self.encoder(x_centered) + self.b_enc)

        # Decode: Map to target space (Layer L or Layer L+1)
        x_hat = self.decoder(acts) + self.b_dec

        if target is not None:
            # Loss: MSE + Sparsity
            l2_loss = F.mse_loss(x_hat, target)
            l1_loss = acts.abs().sum(dim=-1).mean()
            total_loss = l2_loss + self.l1_coeff * l1_loss

            return {
                "output": x_hat,
                "activations": acts,
                "loss": total_loss,
                "l2_loss": l2_loss,
                "l1_loss": l1_loss,
            }

        return {"output": x_hat, "activations": acts}

    @torch.no_grad()
    def normalize_decoder(self):
        """Ensure dictionary atoms are unit norm."""
        W = self.decoder.weight.data
        W.div_(W.norm(dim=0, keepdim=True) + 1e-8)
