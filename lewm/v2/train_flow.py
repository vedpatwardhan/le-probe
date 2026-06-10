import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lewm.v2.velocity_net import ActionVelocityNetwork
from lewm.v2.flow_matcher import ConditionalFlowMatcher, quaternion_to_matrix


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
    dataset=None,
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

    if synthetic:
        print(
            "👾 Generating synthetic trajectories and context with v2 dims (39 proprio)..."
        )
        num_samples = 1000

        # State embedding z_t: (B, embed_dim)
        z_t = torch.randn(num_samples, embed_dim)

        # Proprioception p_t: (B, proprio_dim)
        # We structure p_t specifically:
        # [q(7), dq(7), c_ell(3), r_ell(3), q_e(4), V_ell(1), Kp(7), Kd(7)] -> total 39
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
        if dataset is None:
            raise ValueError(
                "Non-synthetic training requested, but no dataset was provided."
            )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(
        f"🚀 Training Flow Matching | Epochs: {epochs} | Device: {device} | Boundary Lambda: {lambda_c}"
    )

    model.train()

    N_scales = 10  # Number of random reachability map scales to sample per data point

    for epoch in range(epochs):
        epoch_loss = 0.0
        for z_t_b, p_t_b, a1_raw in loader:
            z_t_b = z_t_b.to(device)
            p_t_b = p_t_b.to(device)
            a1_raw = a1_raw.to(device)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_flow_matching_v2(epochs=10, device=device)
