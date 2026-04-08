import torch
import torch.nn as nn
import lightning as pl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import wandb


class MetricsCallback(pl.Callback):
    """
    Advanced interpretability metrics for world model training.
    Monitoring for:
    1. Latent Collapse (SoftRank)
    2. Grounding Linearity (Path Straightening)
    3. Manifold Geometry (PCA Visuals)
    """

    def __init__(self, log_every_n_steps=50, pca_n_samples=256):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.pca_n_samples = pca_n_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps != 0:
            return

        # 'outputs' contains the dict returned by lejepa_forward
        if not isinstance(outputs, dict) or "emb" not in outputs:
            return

        # 1. SoftRank (Effective Rank)
        # emb shape: (B, T, D)
        z = outputs["emb"][:, 0, :]  # Use first timestep for rank analysis
        soft_rank = self.compute_soft_rank(z)
        pl_module.log("research/soft_rank", soft_rank)

        # 2. Path Straightening (Trajectory Linearity)
        # Measures if z_t+1 is 'aligning' in a predictable way
        if z.shape[0] > 1:
            linearity = self.compute_path_straightening(outputs["emb"])
            pl_module.log("research/path_straightening", linearity)

        # 3. PCA Visualization (Cloud Geometry)
        if batch_idx % (self.log_every_n_steps * 4) == 0:
            self.log_pca_to_wandb(z, trainer.current_epoch, batch_idx)

    def compute_soft_rank(self, z):
        """Computes the Effective Rank (SoftRank) of a latent batch."""
        try:
            # Center the latents: Remove the "mean shift" (DC offset)
            # This ensures we measure the dimensionality of the spread, not distance from origin.
            z_centered = z.detach() - z.detach().mean(dim=0, keepdim=True)

            # SVD on the centered latent batch (B, D)
            singular_values = torch.linalg.svdvals(z_centered)

            # Avoid division by zero if all latents are identical/zero
            s_sum = singular_values.sum()
            if s_sum < 1e-12:
                return 1.0
            # Entropy calculation with numerical stability
            dist = singular_values + 1e-10
            entropy = -torch.sum(dist * torch.log(dist))

            return torch.max(torch.tensor(1.0), torch.exp(entropy)).item()
        except Exception as e:
            return 1.0  # Fallback to rank 1 (total collapse) instead of 0

    def compute_path_straightening(self, emb):
        """
        Calculates the average cosine similarity between consecutive
        latent displacement vectors: (z_t - z_t-1) and (z_t+1 - z_t).
        Higher = Straighter/More Predictable trajectories.
        """
        # emb: (B, T, D)
        if emb.shape[1] < 3:
            return 1.0  # Not enough history to compute straightening

        v1 = emb[:, 1] - emb[:, 0]
        v2 = emb[:, 2] - emb[:, 1]

        cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
        return cos_sim.mean().item()

    def log_pca_to_wandb(self, z, epoch, step):
        """Generates and logs a 2D PCA cloud of the latents."""
        z_np = z.detach().cpu().float().numpy()
        if z_np.shape[0] < 2:
            return  # Need at least 2 points to plot/compare

        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_np)

        plt.figure(figsize=(6, 6))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, c="blue")
        plt.title(f"Latent PCA Cloud (Epoch {epoch}, Step {step})")
        plt.xlabel(f"PC1 (Var: {pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 (Var: {pca.explained_variance_ratio_[1]:.2%})")
        plt.grid(True)

        wandb.log({"visuals/latent_pca": wandb.Image(plt)})
        plt.close()
