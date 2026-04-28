import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayerTranscoder(nn.Module):
    """
    Cross-Layer Transcoder (CLT) for mechanistic circuit tracing.
    Maps activations from Layer L to Layer L+1 through a sparse bottleneck.
    """

    def __init__(self, d_model: int, d_sae: int, l1_coeff: float = 0.001):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        # Encoder: Linear + Bias + ReLU (Input is layer L)
        self.encoder = nn.Linear(d_model, d_sae)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: Linear (Output is layer L+1)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x_L, x_L_plus_1=None):
        """
        x_L: Activations from Layer L (Batch, d_model)
        x_L_plus_1: Activations from Layer L+1 (Optional, for training)
        """
        # Centering (usually centered on the target layer's mean or zero)
        x_centered = (
            x_L - self.b_dec
        )  # Note: some implementations center on x_L_plus_1 mean

        # Encode
        acts = F.relu(self.encoder(x_centered) + self.b_enc)

        # Decode (Transcode to Layer L+1)
        x_transcoded = self.decoder(acts) + self.b_dec

        if x_L_plus_1 is not None:
            # Loss: How well does layer L features predict layer L+1 activations?
            l2_loss = F.mse_loss(x_transcoded, x_L_plus_1)
            l1_loss = acts.abs().sum(dim=-1).mean()
            total_loss = l2_loss + self.l1_coeff * l1_loss

            return {
                "transcoded": x_transcoded,
                "activations": acts,
                "loss": total_loss,
                "l2_loss": l2_loss,
                "l1_loss": l1_loss,
            }

        return {"transcoded": x_transcoded, "activations": acts}

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight.data
        W.div_(W.norm(dim=0, keepdim=True) + 1e-8)
