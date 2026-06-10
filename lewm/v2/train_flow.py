import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import stable_pretraining as spt
import argparse
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lewm.v2.velocity_net import ActionVelocityNetwork
from lewm.v2.flow_matcher import ConditionalFlowMatcher, quaternion_to_matrix
from lewm.v2.data import SkeletonDataPluginV2
from lewm.v2.trainer import MultiViewJEPAv2
from lewm.multi_view_encoder import get_multi_view_encoder
from lewm.gr1_modules import GR1Embedder, GR1MLP
from lewm.le_wm.module import ARPredictor
from lewm.skeleton.encoder import patch_vit_for_skeleton
from gr1_config import SCENE_PATH


def add_smooth_trajectory_noise(a1, scale=0.05):
    """
    Applies Gaussian Process (Ornstein-Uhlenbeck) style smooth temporal noise to target trajectory.
    a1: (B, H, D)
    """
    B, H, D = a1.shape
    noise = torch.zeros_like(a1)

    # Generate random walk/smoothed steps over the horizon
    current_noise = torch.randn(B, D, device=a1.device) * scale
    for h in range(H):
        # OU process: dX_t = -theta * X_t * dt + sigma * dW_t
        # Here we approximate with a simple autoregressive dampening parameter (0.8)
        current_noise = (
            0.8 * current_noise + 0.2 * torch.randn(B, D, device=a1.device) * scale
        )
        noise[:, h, :] = current_noise

    return a1 + noise


class FlowMatchingTrainerV2(pl.LightningModule):
    def __init__(
        self,
        horizon=4,
        action_dim=32,
        embed_dim=192,
        proprio_dim=39,
        lr=1e-3,
        lambda_c=0.1,
        synthetic=False,
        ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ActionVelocityNetwork(
            horizon=horizon,
            action_dim=action_dim,
            embed_dim=embed_dim,
            proprio_dim=proprio_dim,
        )
        self.matcher = ConditionalFlowMatcher(sigma=0.0)
        self.model_phase1 = None
        self.synthetic = synthetic
        self.ckpt_path = ckpt_path

    def setup(self, stage=None):
        if not self.synthetic and self.model_phase1 is None:
            if self.ckpt_path is None:
                raise ValueError(
                    "Real dataset training requested, but no Phase 1 checkpoint path (--ckpt_path) was provided."
                )

            print(
                f"🧬 Loading Phase 1 visual encoder from checkpoint: {self.ckpt_path}"
            )

            class MockConfig:
                def __init__(self):
                    self.img_size = 224
                    self.encoder_scale = "tiny"
                    self.patch_size = 14
                    self.use_multi_view = True
                    self.num_views = 5
                    self.fusion_type = "linear"
                    self.predictor = {
                        "hidden_dim": 192,
                        "num_layers": 3,
                        "nhead": 16,
                        "dim_feedforward": 2048,
                        "dropout": 0.1,
                    }
                    self.optimizer = {"lr": 1e-4, "weight_decay": 1e-4}
                    self.wm = {
                        "history_size": 3,
                        "num_preds": 3,
                        "action_dim": 32,
                        "embed_dim": 192,
                    }

                def get(self, key, default=None):
                    return getattr(self, key, default)

            # Load standard pre-trained encoder skeleton model and restore weights
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Strip PyTorch Lightning prefixes if present
            clean_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "") if k.startswith("model.") else k
                clean_state_dict[new_key] = v

            # We temporarily initialize the base multi-view J-EPA and load weights
            cfg = MockConfig()
            encoder = get_multi_view_encoder(cfg)

            # Patch the backbone to 4 channels (RGB + Skeleton) to match the Phase 1 checkpoint
            patch_vit_for_skeleton(encoder.backbone)

            self.model_phase1 = MultiViewJEPAv2(
                encoder=encoder,
                predictor=ARPredictor(
                    num_frames=3,
                    input_dim=192,
                    hidden_dim=192,
                    output_dim=192,
                    depth=3,
                    heads=16,
                    mlp_dim=2048,
                    dropout=0.1,
                ),
                action_encoder=GR1Embedder(input_dim=32, emb_dim=192),
                projector=GR1MLP(input_dim=192, output_dim=192),
                pred_proj=GR1MLP(input_dim=192, output_dim=192),
                use_dino=True,
                fusion_type="linear",
                num_views=5,
            )

            self.model_phase1.load_state_dict(clean_state_dict, strict=False)
            self.model_phase1.eval()

            # Freeze Phase 1 parameters
            for param in self.model_phase1.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        if self.synthetic:
            z_t_b, p_t_b, a1_raw = batch
            J_v = torch.zeros(
                p_t_b.shape[0], 3, 7, device=self.device, dtype=p_t_b.dtype
            )
            q_t_full = torch.zeros(
                p_t_b.shape[0], 32, device=self.device, dtype=p_t_b.dtype
            )
        else:
            # Extract real visual and proprioceptive context on-the-fly
            pixels = batch["pixels"]  # [B, T, V, 4, 224, 224]

            # Forward pass through frozen visual encoder to get visual state latent z_t
            with torch.no_grad():
                # We extract z_t at the start of the planning trajectory (t=0)
                pixels_t = pixels[:, 0].unsqueeze(1)  # [B, 1, V, 4, 224, 224]
                info = {"pixels": pixels_t}
                output = self.model_phase1.encode(info)
                z_t_b = output["emb"].squeeze(1)  # [B, 192]

            # Assemble context vector p_t from the batch:
            # [q_t(7), dq_t(7), c(3), r(3), q_e(4), V(1), kp(7), kd(7)] -> total 39
            q_t = batch["observation.state"][:, 0, 16:23]  # Right arm joints
            dq_t = batch["action"][:, 0, 16:23]

            c = batch["ellipsoid_center"][:, 0]
            r = batch["ellipsoid_radii"][:, 0]
            q_e = batch["ellipsoid_quat"][:, 0]
            V = batch["ellipsoid_volume"][:, 0]

            kp = batch["kp"][:, 0]
            kd = batch["kd"][:, 0]

            p_t_b = torch.cat([q_t, dq_t, c, r, q_e, V, kp, kd], dim=-1)

            # Extract Jacobian of the right arm end-effector
            if "ellipsoid_jacobian" in batch:
                J_v = batch["ellipsoid_jacobian"][:, 0]  # (B, 3, 7)
            else:
                J_v = torch.zeros(
                    batch["observation.state"].shape[0],
                    3,
                    7,
                    device=self.device,
                    dtype=p_t_b.dtype,
                )

            # Target Actions sequence over the horizon
            a1_raw = batch["action"][:, : self.hparams.horizon]  # [B, H, 32]
            q_t_full = batch["observation.state"][:, 0]  # (B, 32)

        N_scales = (
            10  # Number of random reachability map scales to sample per data point
        )
        z_t_b = z_t_b.repeat_interleave(N_scales, dim=0)
        p_t_b = p_t_b.repeat_interleave(N_scales, dim=0)
        a1_raw = a1_raw.repeat_interleave(N_scales, dim=0)
        J_v = J_v.repeat_interleave(N_scales, dim=0)
        q_t_full_repeated = q_t_full.repeat_interleave(N_scales, dim=0)

        # Sample random scale factors S ~ Uniform(0.2, 1.0)
        S = (
            torch.rand(p_t_b.shape[0], 1, device=self.device, dtype=p_t_b.dtype) * 0.8
            + 0.2
        )

        # Extract ellipsoid parameters from p_t
        c = p_t_b[:, 14:17]  # Center (B, 3)
        r = p_t_b[:, 17:20]  # Original Radii (B, 3)
        q_e = p_t_b[:, 20:24]  # Quaternion (B, 4)

        # Scale the ellipsoid radii in the proprioceptive context input
        r_scaled = r * S
        p_t_b[:, 17:20] = r_scaled

        # 2. Transform targets to Relative Delta Actions (a_k - q_t)
        a1_b = a1_raw - q_t_full_repeated.unsqueeze(1)

        # 3. Project target action velocities to stay inside the scaled ellipsoid
        R = quaternion_to_matrix(q_e)  # (B, 3, 3)
        dt = 0.1
        r_scaled_safe = torch.clamp(r_scaled * dt, min=1e-5)

        for k in range(self.hparams.horizon):
            dq_arm = a1_b[:, k, 16:23]  # (B, 7)
            # Map right-arm joint space deltas to end-effector workspace: v_ee = J_v * dq_arm
            v_ee = torch.bmm(J_v, dq_arm.unsqueeze(-1)).squeeze(-1)  # (B, 3)

            delta_v = v_ee - c
            delta_v_local = torch.bmm(
                R.transpose(-1, -2), delta_v.unsqueeze(-1)
            ).squeeze(-1)

            # Compute Mahalanobis distance
            d2 = torch.sum((delta_v_local / r_scaled_safe) ** 2, dim=-1)
            d = torch.sqrt(torch.clamp(d2, min=1e-8))

            # Project if outside the scaled boundaries
            scale_mask = d > 1.0
            if scale_mask.any():
                proj_factor = torch.where(
                    scale_mask, 1.0 / d, torch.ones_like(d)
                ).unsqueeze(-1)
                delta_v_local = delta_v_local * proj_factor
                delta_v_proj = torch.bmm(R, delta_v_local.unsqueeze(-1)).squeeze(-1)
                v_ee_proj = c + delta_v_proj

                # Map back to joint space delta via pseudo-inverse: dq_proj = J_v_pinv * v_ee_proj
                J_v_pinv = torch.linalg.pinv(J_v)  # (B, 7, 3)
                dq_arm_proj = torch.bmm(J_v_pinv, v_ee_proj.unsqueeze(-1)).squeeze(
                    -1
                )  # (B, 7)

                a1_b[:, k, 16:23] = torch.where(
                    scale_mask.unsqueeze(-1), dq_arm_proj, dq_arm
                )

        # 4. Stochastic target perturbation (Smooth GP Action Augmentation)
        a1_b = add_smooth_trajectory_noise(a1_b, scale=0.02)

        # 5. Sample starting Gaussian noise a0
        a0_b = torch.randn_like(a1_b)

        # Compute loss with physics constraint hinge penalty
        loss, cfm_loss, boundary_loss = self.matcher.compute_loss(
            velocity_net=self.model,
            a0=a0_b,
            a1=a1_b,
            z_t=z_t_b,
            p_t=p_t_b,
            J_v=J_v,
            lambda_c=self.hparams.lambda_c,
        )

        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "train/cfm_loss",
            cfm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/boundary_loss",
            boundary_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/scale_factor_S_mean",
            S.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )


def train_flow_matching_v2(
    epochs=100,
    batch_size=256,
    lr=1e-3,
    horizon=8,
    action_dim=32,
    embed_dim=192,
    proprio_dim=39,  # Size matches Kp/Kd and ellipsoid vectors
    device="cpu",
    synthetic=True,
    dataset_name="gr1_pickup_grasp_2k",
    ckpt_path=None,
    lambda_c=0.1,  # Weight on safe ellipsoid boundary constraint
    wandb_project=None,
    wandb_entity=None,
    subdir=None,
    num_workers=4,
):
    """
    Trains ActionVelocityNetwork using v2 Conditional Flow Matching (CFM) wrapped in PyTorch Lightning.
    """
    print(f"⚙️ Setting up Le-Probe v2 Action Flow Matching Training...")

    if synthetic:
        print(
            "👾 Generating synthetic trajectories and context with v2 dims (39 proprio)..."
        )
        num_samples = 1000
        z_t = torch.randn(num_samples, embed_dim)
        p_t = torch.randn(num_samples, proprio_dim)
        p_t[:, 17:20] = torch.clamp(p_t[:, 17:20].abs(), min=0.1)
        p_t[:, 20:24] = F.normalize(p_t[:, 20:24], p=2, dim=-1)
        expert_actions = torch.randn(num_samples, horizon, action_dim)

        dataset = TensorDataset(z_t, p_t, expert_actions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        print(f"📦 Loading real dataset: {dataset_name}...")
        keys_to_load = [
            "observation.state",
            "action",
            "world_center",
            "world_left",
            "world_right",
            "world_top",
            "world_wrist",
        ]
        dataset = SkeletonDataPluginV2(
            repo_id=dataset_name,
            keys_to_load=keys_to_load,
            num_steps=horizon + 1,
            use_virtual_actions=False,
            use_multi_view=True,
            img_size=224,
            use_subset=True,
        )
        # Clear cache before forking workers for memory safety and deadlock prevention
        dataset.clear_cache()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

    # Initialize PyTorch Lightning module
    flow_matching_module = FlowMatchingTrainerV2(
        horizon=horizon,
        action_dim=action_dim,
        embed_dim=embed_dim,
        proprio_dim=proprio_dim,
        lr=lr,
        lambda_c=lambda_c,
        synthetic=synthetic,
        ckpt_path=ckpt_path,
    )

    # Setup logger
    logger = None
    if wandb_project:
        print(f"📊 Initializing Wandb Logger ({wandb_entity}/{wandb_project})...")
        logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=(
                subdir if subdir else f"flow_matching_{dataset_name}_lambda_{lambda_c}"
            ),
        )

    # Configure checkpoint callback
    run_dir = (
        f"./outputs/{subdir}" if subdir else f"./outputs/flow_matching_{dataset_name}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir + "/checkpoints",
        filename="flow_matching-{epoch:02d}-{train/loss:.4f}",
        save_top_k=3,
        monitor="train/loss",
        mode="min",
    )

    # Launch PyTorch Lightning Trainer
    trainer_args = {
        "max_epochs": epochs,
        "accelerator": (
            "gpu" if torch.cuda.is_available() and device != "cpu" else "cpu"
        ),
        "devices": 1,
        "logger": logger,
        "callbacks": [checkpoint_callback],
    }

    trainer = pl.Trainer(**trainer_args)
    print(f"🚀 Launching Flow Matching Training Loop (Batch Size: {batch_size})...")
    trainer.fit(flow_matching_module, train_dataloaders=loader)

    print("✅ Flow Matching v2 Training Completed.")
    return flow_matching_module.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Le-Probe v2 Flow Matching Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=4)  # Match MPC horizon (H=4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic dataset for verify run"
    )
    parser.add_argument("--dataset_name", type=str, default="gr1_pickup_grasp_2k")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Phase 1 J-EPA checkpoint path"
    )
    parser.add_argument(
        "--lambda_c", type=float, default=0.1, help="Boundary loss scale"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Wandb Project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb Entity name"
    )
    parser.add_argument(
        "--subdir", type=str, default=None, help="Output subdirectory name"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker processes"
    )
    args = parser.parse_args()

    train_flow_matching_v2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        horizon=args.horizon,
        device=args.device,
        synthetic=args.synthetic,
        dataset_name=args.dataset_name,
        ckpt_path=args.ckpt_path,
        lambda_c=args.lambda_c,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        subdir=args.subdir,
        num_workers=args.num_workers,
    )
