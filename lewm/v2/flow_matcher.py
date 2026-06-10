import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_matrix(q):
    """
    Converts a batch of unit quaternions (w, x, y, z) to rotation matrices SO(3).
    q: (B, 4)
    Returns: (B, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    R = torch.stack(
        [
            torch.stack(
                [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - zw), 2.0 * (xz + yw)], dim=-1
            ),
            torch.stack(
                [2.0 * (xy + zw), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - xw)], dim=-1
            ),
            torch.stack(
                [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (x2 + y2)], dim=-1
            ),
        ],
        dim=-2,
    )
    return R


class ConditionalFlowMatcher:
    """
    Conditional Flow Matcher (CFM) for Le-Probe v2.
    Translates starting noise trajectories to target trajectories via Optimal Transport paths,
    integrating kinematic capacity (ellipsoid) boundary constraints and relative delta action spaces.
    """

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def sample_path(self, a0, a1):
        """
        Samples virtual times tau ~ Uniform(0,1) and interpolates trajectory states.
        a0: (B, H, D) - Noise
        a1: (B, H, D) - Target Action trajectory
        """
        B, H, D = a0.shape
        device = a0.device
        dtype = a0.dtype

        tau = torch.rand((B, 1), device=device, dtype=dtype)
        tau_expanded = tau.unsqueeze(-1)  # (B, 1, 1)

        # Linear probability path: a_tau = (1 - tau)*a0 + tau*a1
        mean = (1.0 - tau_expanded) * a0 + tau_expanded * a1
        if self.sigma > 0:
            a_tau = mean + self.sigma * torch.randn_like(a0)
        else:
            a_tau = mean

        # Target constant vector field
        target_velocity = a1 - a0
        return tau, a_tau, target_velocity

    def compute_boundary_loss(self, pred_velocity, p_t, J_v=None):
        """
        Computes the Mahalanobis distance of predicted end-effector velocities
        from the Chebyshev ellipsoid boundary.

        p_t: (B, 39) proprioception context.
        J_v: (B, 3, action_dim) Jacobian of the right arm end-effector.
             If not provided, uses a default placeholder identity mapping.
        """
        B = p_t.shape[0]
        device = p_t.device

        # Extract ellipsoid parameters from p_t
        # p_t: [q(7), dq(7), c_ell(3), r_ell(3), q_e(4), V_ell(1), Kp(7), Kd(7)]
        c = p_t[:, 14:17]  # Center (B, 3)
        r = p_t[:, 17:20]  # Semi-axes radii (B, 3)
        q_e = p_t[:, 20:24]  # Unit quaternion (B, 4)

        # Scale velocity limits to position displacement limits over 1 control step (dt = 0.1s)
        dt = 0.1
        r = torch.clamp(r * dt, min=1e-5)

        # We look at the first prediction step (k=0) for boundary compliance
        v_pred_first = pred_velocity[:, 0, :]  # (B, action_dim)

        # Project joint space velocities to end-effector workspace: v_ee = J_v * v_pred
        if J_v is not None:
            # J_v is (B, 3, 7). Extract right arm joints: 16 to 22 (7 DoF)
            v_pred_right = v_pred_first[:, 16:23]  # (B, 7)
            v_ee = torch.bmm(J_v, v_pred_right.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        else:
            # Placeholder/Identity projection mapping for action dims to 3D translational
            v_ee = v_pred_first[:, :3]

        # Convert quaternion to rotation matrix
        R = quaternion_to_matrix(q_e)  # (B, 3, 3)

        # Project relative velocity into the ellipsoid coordinate frame: R^T * (v_ee - c)
        delta_v = v_ee - c
        # (B, 3, 3) x (B, 3, 1) -> (B, 3)
        delta_v_local = torch.bmm(R.transpose(-1, -2), delta_v.unsqueeze(-1)).squeeze(
            -1
        )

        # Compute Mahalanobis distance: sum_i (delta_v_local_i / r_i)^2
        d2 = torch.sum((delta_v_local / r) ** 2, dim=-1)  # (B,)

        # Hinge loss: penalize distance > 1.0 (velocity outside safe ellipsoid boundary)
        boundary_loss = torch.clamp(d2 - 1.0, min=0.0).mean()
        return boundary_loss

    def compute_loss(self, velocity_net, a0, a1, z_t, p_t, J_v=None, lambda_c=0.1):
        """
        Computes conditional flow matching loss + physical ellipsoid boundary penalty.
        """
        tau, a_tau, target_velocity = self.sample_path(a0, a1)
        pred_velocity = velocity_net(a_tau, tau, z_t, p_t)

        cfm_loss = F.mse_loss(pred_velocity, target_velocity)
        boundary_loss = self.compute_boundary_loss(pred_velocity, p_t, J_v)

        total_loss = cfm_loss + lambda_c * boundary_loss
        return total_loss, cfm_loss, boundary_loss

    @torch.no_grad()
    def integrate(
        self, velocity_net, z_t, p_t, dampening=None, steps=8, method="euler"
    ):
        """
        Integrates the learned velocity field to generate planned trajectories.
        """
        B = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype
        H = velocity_net.horizon
        D = velocity_net.action_dim

        a = torch.randn(B, H, D, device=device, dtype=dtype)
        dtau = 1.0 / steps

        if dampening is not None:
            dampening_expanded = dampening.unsqueeze(1)
        else:
            dampening_expanded = 1.0

        if method == "euler":
            for step in range(steps):
                tau = step * dtau
                tau_tensor = torch.full((B, 1), tau, device=device, dtype=dtype)
                v = velocity_net(a, tau_tensor, z_t, p_t)
                a = a + dtau * (dampening_expanded * v)
        elif method == "rk4":
            for step in range(steps):
                tau = step * dtau
                tau_tensor = torch.full((B, 1), tau, device=device, dtype=dtype)

                v1 = velocity_net(a, tau_tensor, z_t, p_t)

                tau_half = tau + 0.5 * dtau
                tau_half_tensor = torch.full(
                    (B, 1), tau_half, device=device, dtype=dtype
                )
                a_half1 = a + 0.5 * dtau * (dampening_expanded * v1)
                v2 = velocity_net(a_half1, tau_half_tensor, z_t, p_t)

                a_half2 = a + 0.5 * dtau * (dampening_expanded * v2)
                v3 = velocity_net(a_half2, tau_half_tensor, z_t, p_t)

                tau_next = tau + dtau
                tau_next_tensor = torch.full(
                    (B, 1), tau_next, device=device, dtype=dtype
                )
                a_next = a + dtau * (dampening_expanded * v3)
                v4 = velocity_net(a_next, tau_next_tensor, z_t, p_t)

                a = a + (dtau / 6.0) * dampening_expanded * (
                    v1 + 2.0 * v2 + 2.0 * v3 + v4
                )
        else:
            raise ValueError(f"Unknown integration method: {method}")

        return a
