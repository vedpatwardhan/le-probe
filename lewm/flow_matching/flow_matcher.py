import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalFlowMatcher:
    """
    Conditional Flow Matcher (CFM) for multi-step action trajectories.

    Translates random noise action sequences a_0 ~ N(0, I) to expert target
    action sequences a_1 via a continuous-time linear probability path.
    """

    def __init__(self, sigma=0.0):
        """
        sigma: standard deviation of conditional path noise (0.0 for deterministic OT)
        """
        self.sigma = sigma

    def sample_path(self, a0, a1):
        """
        Constructs the conditional probability path and samples time steps.

        a0: (B, horizon, action_dim) - start noise trajectory (tau = 0)
        a1: (B, horizon, action_dim) - end target action sequence (tau = 1)

        Returns:
            tau: (B, 1) - sampled virtual times tau ~ Uniform(0, 1)
            a_tau: (B, horizon, action_dim) - action sequence along the path at tau
            target_velocity: (B, horizon, action_dim) - target velocity vector u(a_tau)
        """
        B, H, D = a0.shape
        device = a0.device
        dtype = a0.dtype

        # 1. Sample tau uniformly for each sample in the batch
        tau = torch.rand((B, 1), device=device, dtype=dtype)

        # Reshape tau to broadcast across sequence dimensions (B, 1, 1)
        tau_expanded = tau.unsqueeze(-1)

        # 2. Linear path interpolation: a_tau = (1 - tau) * a0 + tau * a1
        mean = (1.0 - tau_expanded) * a0 + tau_expanded * a1

        # Add path noise if sigma > 0
        if self.sigma > 0:
            noise = torch.randn_like(a0)
            a_tau = mean + self.sigma * noise
        else:
            a_tau = mean

        # 3. Target velocity: da/dtau = a1 - a0
        target_velocity = a1 - a0

        return tau, a_tau, target_velocity

    def compute_loss(self, velocity_net, a0, a1, z_t, z_bar, p_t):
        """
        Computes the flow matching MSE loss for a batch of trajectories.
        """
        tau, a_tau, target_velocity = self.sample_path(a0, a1)
        pred_velocity = velocity_net(a_tau, tau, z_t, z_bar, p_t)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def integrate(
        self, velocity_net, z_t, z_bar, p_t, dampening=None, steps=8, method="euler"
    ):
        """
        Solves the ODE da/dtau = v(a, tau | z_t, z_bar, p_t) from tau = 0 to tau = 1
        to generate the planned action trajectory.

        z_t: (B, embed_dim) - current state embedding
        z_bar: (B, embed_dim) - subgoal target visual waypoint
        p_t: (B, proprio_dim) - current proprioceptive and torque capacities
        dampening: (B, action_dim) - dimension-wise dampening coefficients D(p_t) in [0, 1]
        steps: number of ODE integration steps
        method: ODE solver integration method ('euler' or 'rk4')
        """
        B = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype
        H = velocity_net.horizon
        D = velocity_net.action_dim

        # Start from Gaussian noise
        a = torch.randn(B, H, D, device=device, dtype=dtype)
        dtau = 1.0 / steps

        # Expand dampening vector to match sequence steps if provided
        # dampening is expected to be shape (B, action_dim)
        if dampening is not None:
            # Shape becomes (B, 1, action_dim) for broadcasting over horizon H
            dampening_expanded = dampening.unsqueeze(1)
        else:
            dampening_expanded = 1.0

        if method == "euler":
            for step in range(steps):
                tau = step * dtau
                tau_tensor = torch.full((B, 1), tau, device=device, dtype=dtype)

                # Get raw model velocity
                v = velocity_net(a, tau_tensor, z_t, z_bar, p_t)

                # Apply time dampening step
                a = a + dtau * (dampening_expanded * v)

        elif method == "rk4":
            for step in range(steps):
                tau = step * dtau
                tau_tensor = torch.full((B, 1), tau, device=device, dtype=dtype)

                # k1
                v1 = velocity_net(a, tau_tensor, z_t, z_bar, p_t)

                # k2
                tau_half = tau + 0.5 * dtau
                tau_half_tensor = torch.full(
                    (B, 1), tau_half, device=device, dtype=dtype
                )
                a_half1 = a + 0.5 * dtau * (dampening_expanded * v1)
                v2 = velocity_net(a_half1, tau_half_tensor, z_t, z_bar, p_t)

                # k3
                a_half2 = a + 0.5 * dtau * (dampening_expanded * v2)
                v3 = velocity_net(a_half2, tau_half_tensor, z_t, z_bar, p_t)

                # k4
                tau_next = tau + dtau
                tau_next_tensor = torch.full(
                    (B, 1), tau_next, device=device, dtype=dtype
                )
                a_next = a + dtau * (dampening_expanded * v3)
                v4 = velocity_net(a_next, tau_next_tensor, z_t, z_bar, p_t)

                # Update using weighted average of velocities
                a = a + (dtau / 6.0) * dampening_expanded * (
                    v1 + 2.0 * v2 + 2.0 * v3 + v4
                )

        else:
            raise ValueError(f"Unknown integration method: {method}")

        return a
