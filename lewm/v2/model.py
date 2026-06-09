import torch
import torch.nn as nn
from lewm.gr1_modules import MultiViewJEPA


class ValuePredictor(nn.Module):
    """
    Goal-conditioned Value Head.
    Predicts the cumulative future rewards V(z_t | z_bar) from the current latent state z_t
    and the fused subgoal waypoint representation z_bar.
    """

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_t, z_bar):
        # z_t: (..., D)
        # z_bar: (..., D)
        x = torch.cat([z_t, z_bar], dim=-1)
        return self.net(x)


class MultiViewJEPAv2(MultiViewJEPA):
    """
    World-Action Interactive Model (WAIM) / Le-Probe v2 model class.
    Extends MultiViewJEPA with a goal-conditioned Value Head.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = (
            self.dino_projector.net[-1].out_features
            if hasattr(self, "dino_projector")
            else kwargs.get("embed_dim", 192)
        )
        # Value Head input is concatenated z_t (embed_dim) and z_bar (embed_dim)
        self.value_head = ValuePredictor(input_dim=embed_dim * 2, hidden_dim=512)
