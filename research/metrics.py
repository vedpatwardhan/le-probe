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

        # 1. Latent Diagnostics (Rank and Variance)
        # emb shape: (B, T, D)
        z = outputs["emb"][:, 0, :]  # Use first timestep for analysis
        diagnostics = self.compute_latent_diagnostics(z)

        pl_module.log("research/soft_rank", diagnostics["soft_rank"])
        pl_module.log(
            "research/participation_ratio", diagnostics["participation_ratio"]
        )

        # 2. Path Straightening (Trajectory Linearity)
        # Measures if z_t+1 is 'aligning' in a predictable way
        if z.shape[0] > 1:
            linearity = self.compute_path_straightening(outputs["emb"])
            pl_module.log("research/path_straightening", linearity)

        # 3. PCA Visualization (Cloud Geometry)
        if batch_idx % (self.log_every_n_steps * 4) == 0:
            self.log_pca_to_wandb(z, trainer.current_epoch, batch_idx)

    @staticmethod
    def compute_latent_diagnostics(z):
        """
        Performs a single SVD and returns both SoftRank (Entropy)
        and Participation Ratio in a dictionary.
        """
        try:
            # emb: (B, D)
            if z.shape[0] < 2:
                return {"soft_rank": 1.0, "participation_ratio": 1.0}

            # Center the latents
            z_centered = z.detach() - z.detach().mean(dim=0, keepdim=True)

            # SVD on the centered latent batch (B, D)
            singular_values = torch.linalg.svdvals(z_centered)

            # Avoid division by zero
            s_sum = singular_values.sum()
            if s_sum < 1e-16:
                return {"soft_rank": 1.0, "participation_ratio": 1.0}

            # 1. Spectral Entropy Rank (Diversity focused)
            p = singular_values / s_sum
            entropy = -torch.sum(p * torch.log(p + 1e-12))
            soft_rank = torch.exp(entropy).item()

            # 2. Participation Ratio Rank (Variance focused)
            lambdas = singular_values**2
            pr_rank = (lambdas.sum() ** 2) / (lambdas**2).sum()

            return {
                "soft_rank": max(1.0, soft_rank),
                "participation_ratio": max(1.0, pr_rank.item()),
            }
        except Exception as e:
            return {"soft_rank": 1.0, "participation_ratio": 1.0}

    @staticmethod
    def compute_path_straightening(emb):
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
