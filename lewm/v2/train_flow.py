import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import stable_pretraining as spt
import argparse

from lewm.v2.velocity_net import ActionVelocityNetwork
from lewm.v2.flow_matcher import ConditionalFlowMatcher, quaternion_to_matrix
from lewm.v2.data import SkeletonDataPluginV2
from lewm.v2.trainer import MultiViewJEPAv2
from lewm.multi_view_encoder import get_multi_view_encoder
from lewm.gr1_modules import ARPredictor, GR1Embedder, GR1MLP
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
):
    """
    Trains ActionVelocityNetwork using v2 Conditional Flow Matching (CFM).
    Integrates relative delta action space conversion, target trajectory augmentation,
    and capacity boundary constraint matching.
    """
    print(f"⚙️ Setting up Le-Probe v2 Action Flow Matching Training...")

    model = ActionVelocityNetwork(
        horizon=horizon,
        action_dim=action_dim,
        embed_dim=embed_dim,
        proprio_dim=proprio_dim,
    ).to(device)

    matcher = ConditionalFlowMatcher(sigma=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model_phase1 = None

    if synthetic:
        print(
            "👾 Generating synthetic trajectories and context with v2 dims (39 proprio)..."
        )
        num_samples = 1000

        # State embedding z_t: (B, embed_dim)
        z_t = torch.randn(num_samples, embed_dim)

        # Proprioception p_t: (B, proprio_dim)
        p_t = torch.randn(num_samples, proprio_dim)
        # Ensure radii are strictly positive
        p_t[:, 17:20] = torch.clamp(p_t[:, 17:20].abs(), min=0.1)
        # Ensure quaternion is normalized
        p_t[:, 20:24] = F.normalize(p_t[:, 20:24], p=2, dim=-1)

        # Raw target expert actions (absolute positions)
        expert_actions = torch.randn(num_samples, horizon, action_dim)

        ds = TensorDataset(z_t, p_t, expert_actions)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
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

        # Instantiate dataset (will load the cached data including pre-computed ellipsoids)
        dataset = SkeletonDataPluginV2(
            repo_id=dataset_name,
            keys_to_load=keys_to_load,
            num_steps=horizon + 1,  # Need enough steps for planning horizon
            use_virtual_actions=False,
            use_multi_view=True,
            img_size=224,
            use_subset=True,
        )

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )

        # Load Phase 1 J-EPA model for visual embedding extraction
        if ckpt_path is None:
            raise ValueError(
                "Real dataset training requested, but no Phase 1 checkpoint path (--ckpt_path) was provided."
            )

        print(f"🧬 Loading Phase 1 visual encoder from checkpoint: {ckpt_path}")

        # Mocks to allow instantiation without complex hydra configs
        class MockConfig:
            def __init__(self):
                self.img_size = 224
                self.predictor = {
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "nhead": 4,
                    "dim_feedforward": 512,
                    "dropout": 0.1,
                }
                self.optimizer = {"lr": 1e-4, "weight_decay": 1e-4}
                self.wm = {
                    "history_size": 3,
                    "num_preds": 3,
                    "action_dim": 32,
                    "embed_dim": 192,
                }

        # Load standard pre-trained encoder skeleton model and restore weights
        checkpoint = torch.load(ckpt_path, map_location="cpu")
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
        from lewm.skeleton.encoder import patch_vit_for_skeleton

        patch_vit_for_skeleton(encoder.backbone)

        model_phase1 = MultiViewJEPAv2(
            encoder=encoder,
            predictor=ARPredictor(
                num_frames=3, input_dim=192, hidden_dim=256, output_dim=256
            ),
            action_encoder=GR1Embedder(input_dim=32, emb_dim=192),
            projector=GR1MLP(input_dim=256, output_dim=192),
            pred_proj=GR1MLP(input_dim=256, output_dim=192),
            use_dino=True,
            fusion_type="linear",
            num_views=5,
        )

        model_phase1.load_state_dict(clean_state_dict, strict=False)
        model_phase1 = model_phase1.to(device)
        model_phase1.eval()

    print(
        f"🚀 Training Flow Matching | Epochs: {epochs} | Device: {device} | Boundary Lambda: {lambda_c}"
    )

    model.train()

    N_scales = 10  # Number of random reachability map scales to sample per data point

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            if synthetic:
                z_t_b, p_t_b, a1_raw = batch
                z_t_b = z_t_b.to(device)
                p_t_b = p_t_b.to(device)
                a1_raw = a1_raw.to(device)
            else:
                # Extract real visual and proprioceptive context on-the-fly
                pixels = batch["pixels"].to(device)  # [B, T, V, 4, 224, 224]

                # Forward pass through frozen visual encoder to get visual state latent z_t
                with torch.no_grad():
                    # We extract z_t at the start of the planning trajectory (t=0)
                    pixels_t = pixels[:, 0].unsqueeze(1)  # [B, 1, V, 4, 224, 224]
                    info = {"pixels": pixels_t}
                    output = model_phase1.encode(info)
                    z_t_b = output["emb"].squeeze(1)  # [B, 192]

                # Assemble context vector p_t from the batch:
                # [q_t(7), dq_t(7), c(3), r(3), q_e(4), V(1), kp(7), kd(7)] -> total 39
                q_t = batch["observation.state"][:, 0, 16:23].to(
                    device
                )  # Right arm joints
                # Extract actions as an approximation of velocity/next targets
                dq_t = batch["action"][:, 0, 16:23].to(device)

                c = batch["ellipsoid_center"][:, 0].to(device)
                r = batch["ellipsoid_radii"][:, 0].to(device)
                q_e = batch["ellipsoid_quat"][:, 0].to(device)
                V = batch["ellipsoid_volume"][:, 0].to(device)

                kp = batch["kp"][:, 0].to(device)
                kd = batch["kd"][:, 0].to(device)

                p_t_b = torch.cat([q_t, dq_t, c, r, q_e, V, kp, kd], dim=-1)

                # Target Actions sequence over the horizon
                a1_raw = batch["action"][:, :horizon].to(device)  # [B, H, 32]

            # 1. Multi-scale replication (N=10) to teach the model how to scale actions
            B_orig = z_t_b.shape[0]
            z_t_b = z_t_b.repeat_interleave(N_scales, dim=0)
            p_t_b = p_t_b.repeat_interleave(N_scales, dim=0)
            a1_raw = a1_raw.repeat_interleave(N_scales, dim=0)

            # Sample random scale factors S ~ Uniform(0.2, 1.0)
            S = (
                torch.rand(p_t_b.shape[0], 1, device=device, dtype=p_t_b.dtype) * 0.8
                + 0.2
            )

            # Extract ellipsoid parameters from p_t
            # p_t: [q(7), dq(7), c_ell(3), r_ell(3), q_e(4), V_ell(1), Kp(7), Kd(7)]
            c = p_t_b[:, 14:17]  # Center (B, 3)
            r = p_t_b[:, 17:20]  # Original Radii (B, 3)
            q_e = p_t_b[:, 20:24]  # Quaternion (B, 4)

            # Scale the ellipsoid radii in the proprioceptive context input
            r_scaled = r * S
            p_t_b[:, 17:20] = r_scaled

            # 2. Transform targets to Relative Delta Actions (a_k - q_t)
            q_t_b = p_t_b[:, 0:7]  # (B, 7)
            q_t_padded = torch.zeros(
                q_t_b.shape[0], action_dim, device=device, dtype=q_t_b.dtype
            )
            q_t_padded[:, :7] = q_t_b
            a1_b = a1_raw - q_t_padded.unsqueeze(1)

            # 3. Project target action velocities to stay inside the scaled ellipsoid
            R = quaternion_to_matrix(q_e)  # (B, 3, 3)
            r_scaled_safe = torch.clamp(r_scaled, min=1e-5)

            for k in range(horizon):
                # We project the first 3 action dims (translational EE motion)
                v_k = a1_b[:, k, :3]
                delta_v = v_k - c

                # Project delta velocity into local ellipsoid coordinate system
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

                    # Reconstruct global projected velocity
                    delta_v_proj = torch.bmm(R, delta_v_local.unsqueeze(-1)).squeeze(-1)
                    a1_b[:, k, :3] = c + delta_v_proj

            # 4. Stochastic target perturbation (Smooth GP Action Augmentation)
            a1_b = add_smooth_trajectory_noise(a1_b, scale=0.02)

            # 5. Sample starting Gaussian noise a0
            a0_b = torch.randn_like(a1_b)

            optimizer.zero_grad()

            # Compute loss with physics constraint hinge penalty
            loss = matcher.compute_loss(
                velocity_net=model,
                a0=a0_b,
                a1=a1_b,
                z_t=z_t_b,
                p_t=p_t_b,
                lambda_c=lambda_c,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs:03d} | Loss: {avg_loss:.6f}")

    print("✅ Flow Matching v2 Training Completed.")
    return model


if __name__ == "__main__":
    import argparse

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
    )
