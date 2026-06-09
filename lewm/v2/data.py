import torch
import numpy as np
from lewm.skeleton.data import SkeletonDataPlugin


class SkeletonDataPluginV2(SkeletonDataPlugin):
    """
    SkeletonDataPluginV2 implements the data pipeline for Le-Probe v2.
    It returns joint-level controller gains (Kp, Kd) as part of proprioception
    and provides a composite metric for cross-episode hindsight target retrieval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default gains: GR1 typical values or standardized defaults
        # If dataset lacks Kp/Kd columns, we mock them to be compatible
        self.default_kp = torch.ones(7) * 100.0  # Stiffness
        self.default_kd = torch.ones(7) * 10.0  # Damping

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        # 1. Inject controller gains telemetry into the batch
        # Extract or default Kp, Kd
        B_steps = self.num_steps
        kp = self.default_kp.unsqueeze(0).repeat(B_steps, 1)  # [T, 7]
        kd = self.default_kd.unsqueeze(0).repeat(B_steps, 1)  # [T, 7]

        # In a real setup, we would try to retrieve observation.controller_gains from H5/Parquet:
        if "observation.controller_gains" in batch:
            gains = batch["observation.controller_gains"]
            # Split into Kp and Kd
            kp = gains[..., :7]
            kd = gains[..., 7:]

        batch["kp"] = kp
        batch["kd"] = kd

        return batch

    def retrieve_hindsight_target(
        self, z_t, q_t, dq_t, successful_database, weights=(1.0, 1.0, 1.0)
    ):
        """
        Performs Cross-Episode Hindsight Target Retrieval using a composite distance metric.

        z_t: (D,) or (B, D) current visual state latent
        q_t: (7,) or (B, 7) current joint angles
        dq_t: (7,) or (B, 7) current joint velocities
        successful_database: A dictionary/list containing pre-indexed successful states and trajectories.
        weights: (w_z, w_q, w_dq) weighting factors
        """
        w_z, w_q, w_dq = weights
        best_idx = -1
        min_dist = float("inf")

        # Convert to tensor if numpy
        z_t = torch.as_tensor(z_t)
        q_t = torch.as_tensor(q_t)
        dq_t = torch.as_tensor(dq_t)

        # successful_database contains keys: 'z', 'q', 'dq', 'actions'
        # z: [N_success, D]
        # q: [N_success, 7]
        # dq: [N_success, 7]
        db_z = torch.as_tensor(successful_database["z"], device=z_t.device)
        db_q = torch.as_tensor(successful_database["q"], device=q_t.device)
        db_dq = torch.as_tensor(successful_database["dq"], device=dq_t.device)

        # Compute batched composite distance
        # Dist = w_z * ||z_t - z^*||^2 + w_q * ||q_t - q^*||^2 + w_dq * ||dq_t - dq^*||^2
        dist_z = torch.sum((db_z - z_t.unsqueeze(0)) ** 2, dim=-1)
        dist_q = torch.sum((db_q - q_t.unsqueeze(0)) ** 2, dim=-1)
        dist_dq = torch.sum((db_dq - dq_t.unsqueeze(0)) ** 2, dim=-1)

        total_dist = w_z * dist_z + w_q * dist_q + w_dq * dist_dq
        best_idx = torch.argmin(total_dist).item()

        # Retrieve the corresponding action sequence
        target_actions = successful_database["actions"][
            best_idx
        ]  # Shape: [H, action_dim]
        return target_actions, best_idx
