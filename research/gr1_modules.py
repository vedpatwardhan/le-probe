import torch
import torch.nn as nn
import torch.nn.functional as F


class GR1Embedder(nn.Module):
    """
    Robust action encoder with residual connections for GR-1.
    Designed to prevent the 32-DoF signal from being 'crushed'
    by high-magnitude vision embeddings.
    """

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=256,
        emb_dim=10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, smoothed_dim)

        # Residual Block
        self.residual_net = nn.Sequential(
            nn.LayerNorm(smoothed_dim),
            nn.Linear(smoothed_dim, smoothed_dim * 2),
            nn.GELU(),
            nn.Linear(smoothed_dim * 2, smoothed_dim),
            nn.Dropout(0.05),
        )

        self.output_proj = nn.Linear(smoothed_dim, emb_dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        # Project raw actions to hidden space
        h = self.input_proj(x)

        # Apply residual transformation
        h = h + self.residual_net(h)

        # Project to final embedding dimension
        return self.output_proj(h)


class GR1MLP(nn.Module):
    """
    GR-1 Specific MLP with LayerNorm instead of BatchNorm.
    LayerNorm is instance-wise, preventing the 'Symmetry Trap' where
    BatchNorm centers zero-variance batches, hiding collapse from SIGReg.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)
