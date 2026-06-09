import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embedding for the virtual flow matching step tau."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ActionVelocityNetwork(nn.Module):
    """
    Flow Transformer vector field network representing the action denoiser:
    v(a(tau), tau | z_t, p_t)

    Includes conditioning on visual state z_t and proprioception state p_t
    (which contains joint telemetry, ellipsoid bounds, and joint gains).
    """

    def __init__(
        self,
        horizon=8,
        action_dim=32,
        embed_dim=192,
        proprio_dim=39,  # Default to 39 for v2 proprioception (7 q + 7 dq + 11 ellipsoid + 14 gains)
        time_embed_dim=64,
        hidden_dim=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        # Time Embedding for virtual step tau
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # Context projection: projects z_t, p_t, and time embedding into a single context token
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim + proprio_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Action step projection: projects individual action vectors in the sequence to hidden_dim
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Learnable temporal positional embeddings for the action sequence
        self.pos_embedding = nn.Parameter(torch.zeros(1, horizon, hidden_dim))

        # Transformer Encoder Trunk
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Velocity projection: maps hidden tokens back to action velocity predictions
        self.velocity_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, a_tau, tau, z_t, p_t):
        """
        a_tau: (B, horizon, action_dim) - action trajectory coordinates
        tau: (B, 1) or (B,) - virtual temporal flow step in [0, 1]
        z_t: (B, embed_dim) - current latent visual state representation
        p_t: (B, proprio_dim) - proprioception context (telemetry + capacity + gains)
        """
        B = a_tau.shape[0]

        if a_tau.ndim == 2:
            a_tau = a_tau.reshape(B, self.horizon, self.action_dim)

        if tau.ndim == 2:
            tau = tau.squeeze(-1)

        # 1. Project virtual flow time
        t_emb = self.time_embed(tau)  # (B, time_embed_dim)

        # 2. Concat and project context conditions to a single context token
        context = torch.cat([z_t, p_t, t_emb], dim=-1)
        context_token = self.context_projection(context).unsqueeze(
            1
        )  # (B, 1, hidden_dim)

        # 3. Project action steps and add positional embeddings
        action_tokens = self.action_projection(a_tau)  # (B, horizon, hidden_dim)
        action_tokens = action_tokens + self.pos_embedding

        # 4. Concat context token and action tokens: [context, action_1, ..., action_H]
        sequence = torch.cat(
            [context_token, action_tokens], dim=1
        )  # (B, horizon + 1, hidden_dim)

        # 5. Pass through Transformer Encoder
        out_sequence = self.transformer(sequence)  # (B, horizon + 1, hidden_dim)

        # 6. Extract action tokens (skipping the prepended context token) and project to velocities
        out_action_tokens = out_sequence[:, 1:, :]  # (B, horizon, hidden_dim)
        pred_velocities = self.velocity_projection(
            out_action_tokens
        )  # (B, horizon, action_dim)

        return pred_velocities
