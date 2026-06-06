import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lewm.flow_matching.velocity_net import ActionVelocityNetwork
from lewm.flow_matching.flow_matcher import ConditionalFlowMatcher


def train_flow_matching(
    epochs=100,
    batch_size=256,
    lr=1e-3,
    horizon=8,
    action_dim=32,
    embed_dim=192,
    proprio_dim=10,
    device="cpu",
    synthetic=True,
    dataset=None,
):
    """
    Trains the ActionVelocityNetwork using Conditional Flow Matching.

    If synthetic=True, generates a dummy dataset to test execution and sanity.
    Otherwise, trains on the provided dataset of (z_t, z_bar, p_t, expert_actions).
    """
    print(f"⚙️ Setting up Action Flow Matching Training...")

    # 1. Model & Matcher Setup
    model = ActionVelocityNetwork(
        horizon=horizon,
        action_dim=action_dim,
        embed_dim=embed_dim,
        proprio_dim=proprio_dim,
    ).to(device)

    matcher = ConditionalFlowMatcher(sigma=0.0)  # OT flow matching
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 2. Data Preparation
    if synthetic:
        print("👾 Generating synthetic trajectories and contexts for verification...")
        num_samples = 1000

        # Inputs: current state, target waypoint, and proprioceptive features
        z_t = torch.randn(num_samples, embed_dim)
        z_bar = torch.randn(num_samples, embed_dim)
        p_t = torch.randn(num_samples, proprio_dim)

        # Targets: Expert action sequences
        expert_actions = torch.randn(num_samples, horizon, action_dim)

        ds = TensorDataset(z_t, z_bar, p_t, expert_actions)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    else:
        if dataset is None:
            raise ValueError(
                "Non-synthetic training requested, but no dataset was provided."
            )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"🚀 Training Flow Matching | Epochs: {epochs} | Device: {device}")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for z_t_b, z_bar_b, p_t_b, a1_b in loader:
            z_t_b, z_bar_b = z_t_b.to(device), z_bar_b.to(device)
            p_t_b, a1_b = p_t_b.to(device), a1_b.to(device)

            # Sample random initial noise a0
            a0_b = torch.randn_like(a1_b)

            optimizer.zero_grad()
            loss = matcher.compute_loss(model, a0_b, a1_b, z_t_b, z_bar_b, p_t_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs:03d} | Loss: {avg_loss:.6f}")

    print("✅ Flow Matching Training Completed.")
    return model


if __name__ == "__main__":
    # Test execution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_flow_matching(epochs=10, device=device)
