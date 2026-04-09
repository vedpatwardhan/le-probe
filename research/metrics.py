import torch
import torch.nn as nn
import lightning as pl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import wandb
import csv
import os
from datetime import datetime


class MetricsCallback(pl.Callback):
    """
    Advanced interpretability metrics for world model training.
    Monitoring for:
    1. Latent Collapse (SoftRank)
    2. Grounding Linearity (Path Straightening)
    3. Manifold Geometry (PCA Visuals)
    """

    def __init__(
        self,
        log_every_n_steps=50,
        pca_n_samples=256,
        csv_path="manifold_diagnostics.csv",
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.pca_n_samples = pca_n_samples
        self.csv_path = csv_path
        self._init_csv()

    def _init_csv(self):
        """Initializes the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            headers = [
                "timestamp",
                "step",
                "epoch",
                "soft_rank",
                "participation_ratio",
                "action_mag",
                "emb_mag",
                "signal_ratio",
                "path_straightening",
            ]
            # Add headers for top 10 singular values
            headers += [f"s_{i}" for i in range(10)]
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_to_csv(self, data_dict):
        """Appends a row of data to the CSV file."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # Ensure we follow the header order
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data_dict.get("step", 0),
                data_dict.get("epoch", 0),
                f"{data_dict.get('soft_rank', 0):.4f}",
                f"{data_dict.get('participation_ratio', 0):.4f}",
                f"{data_dict.get('action_mag', 0):.6f}",
                f"{data_dict.get('emb_mag', 0):.6f}",
                f"{data_dict.get('signal_ratio', 0):.4f}",
                f"{data_dict.get('path_straightening', 0):.4f}",
            ]
            # singular values
            s_vals = data_dict.get("singular_values", [])
            for i in range(10):
                val = s_vals[i] if i < len(s_vals) else 0.0
                row.append(f"{val:.6f}")
            writer.writerow(row)

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
        linearity = 1.0
        if outputs["emb"].shape[1] > 1:
            linearity = self.compute_path_straightening(outputs["emb"])
            pl_module.log("research/path_straightening", linearity)

        # 3. Action Signal Metrics
        # batch['action'] shape: (B, T, 32)
        act_mag = batch["action"].abs().mean().item()
        emb_mag = z.abs().mean().item()
        sig_ratio = act_mag / (emb_mag + 1e-8)

        pl_module.log("research/action_mag", act_mag)
        pl_module.log("research/signal_ratio", sig_ratio)

        # 4. Persistent CSV Logging
        csv_data = {
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "soft_rank": diagnostics["soft_rank"],
            "participation_ratio": diagnostics["participation_ratio"],
            "action_mag": act_mag,
            "emb_mag": emb_mag,
            "signal_ratio": sig_ratio,
            "path_straightening": linearity,
            "singular_values": diagnostics["singular_values"],
        }
        self.log_to_csv(csv_data)

        # 5. Console Health Check (Immediate Feedback)
        if trainer.global_step == 1:
            print("\n" + "=" * 50)
            print("🩺 LEWM INITIAL HEALTH CHECK (STEP 1)")
            print(f"  - SoftRank:        {diagnostics['soft_rank']:.4f}")
            print(f"  - Participation:   {diagnostics['participation_ratio']:.4f}")
            print(f"  - Action Magnitude: {act_mag:.6f}")
            print(f"  - Embedding Mag:   {emb_mag:.6f}")
            print(f"  - Signal Ratio:    {sig_ratio:.4f}")
            print(
                f"  - Spectral Gap (S0/S1): {diagnostics['singular_values'][0]/(diagnostics['singular_values'][1]+1e-8):.2f}"
            )

            if (
                diagnostics["soft_rank"] < 1.05
                and diagnostics["singular_values"][1] < 1e-6
            ):
                print(
                    "🚨 ALERT: Model is CURRENTLY COLLAPSED. Monitor SigReg gradients."
                )
            else:
                print("✅ MANIFOLD IS BREATHING: Non-zero variance detected.")
            print("=" * 50 + "\n")

        # 6. PCA Visualization (Cloud Geometry)
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
                return {
                    "soft_rank": 1.0,
                    "participation_ratio": 1.0,
                    "singular_values": [0.0] * 10,
                }

            # Center the latents
            z_centered = z.detach() - z.detach().mean(dim=0, keepdim=True)

            # SVD on the centered latent batch (B, D)
            singular_values = torch.linalg.svdvals(z_centered)

            # Avoid division by zero
            s_sum = singular_values.sum()
            if s_sum < 1e-16:
                return {
                    "soft_rank": 1.0,
                    "participation_ratio": 1.0,
                    "singular_values": [0.0] * 10,
                }

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
                "singular_values": singular_values[:10].tolist(),
            }
        except Exception as e:
            return {
                "soft_rank": 1.0,
                "participation_ratio": 1.0,
                "singular_values": [0.0] * 10,
            }

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
