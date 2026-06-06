import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal time embedding for the virtual flow matching step tau.
    """

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
    Continuous vector field MLP representing the action denoiser velocity network:
    v(a(tau), tau | z_t, z_bar, p_t)

    Maps: (action_trajectory, tau, z_t, z_bar, p_t) -> velocity vector da/dtau
    """

    def __init__(
        self,
        horizon=8,
        action_dim=32,
        embed_dim=192,
        proprio_dim=10,
        time_embed_dim=64,
        hidden_dim=256,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.flat_action_dim = horizon * action_dim

        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # Context encoder/projection
        self.context_projection = nn.Sequential(
            nn.Linear(embed_dim * 2 + proprio_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Velocity network trunk
        self.net = nn.Sequential(
            nn.Linear(self.flat_action_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.flat_action_dim),
        )

    def forward(self, a_tau, tau, z_t, z_bar, p_t):
        """
        a_tau: (B, horizon, action_dim) or (B, flat_action_dim)
        tau: (B, 1) or (B,) - virtual temporal flow step in [0, 1]
        z_t: (B, embed_dim) - current latent visual state representation
        z_bar: (B, embed_dim) - target subgoal waypoint representation
        p_t: (B, proprio_dim) - proprioception and physics constraints (torques, etc.)
        """
        B = a_tau.shape[0]

        # Flatten action if input is 3D
        if a_tau.ndim == 3:
            a_tau = a_tau.reshape(B, -1)

        if tau.ndim == 2:
            tau = tau.squeeze(-1)  # (B,)

        # 1. Project virtual flow time
        t_emb = self.time_embed(tau)  # (B, time_embed_dim)

        # 2. Concat and project context conditions
        context = torch.cat([z_t, z_bar, p_t, t_emb], dim=-1)
        context_emb = self.context_projection(context)  # (B, hidden_dim)

        # 3. Concatenate action trajectory with context and pass through trunk
        inp = torch.cat([a_tau, context_emb], dim=-1)
        out = self.net(inp)  # (B, flat_action_dim)

        # Reshape velocity back to trajectory format: (B, horizon, action_dim)
        return out.reshape(B, self.horizon, self.action_dim)
