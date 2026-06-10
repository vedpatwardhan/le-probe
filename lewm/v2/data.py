import torch
import numpy as np
import mujoco
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler
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

    def _init_mujoco(self):
        """Lazy initialization of MuJoCo engine to keep dataloader thread-safe and process-safe."""
        if hasattr(self, "model"):
            return

        # Load model and initialize data
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data = mujoco.MjData(self.model)

        # Right Arm Joints
        self.RIGHT_ARM_JOINTS = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ]
        self.ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link"
        )

        # Gather DOF indices in MuJoCo qpos/qvel layout
        self.dof_indices = []
        for name in self.RIGHT_ARM_JOINTS:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                self.dof_indices.append(self.model.jnt_dofadr[j_id])

    def compute_ellipsoid(self, q_state):
        """
        Computes the Chebyshev/Manipulability ellipsoid parameters from normalized joint states.
        Uses fast analytical SVD of the translational Jacobian.
        """
        self._init_mujoco()

        # 1. Unscale normalized state [-1, 1] back to raw physical radians
        scaler = StandardScaler()
        raw_state = scaler.unscale_action(
            q_state.numpy() if torch.is_tensor(q_state) else q_state
        )

        # 2. Extract right arm joint positions (indices 16 to 22 in wire protocol)
        q_arm = raw_state[16:23]

        # 3. Set robot joint angles and run forward kinematics
        self.data.qpos[self.dof_indices] = q_arm
        mujoco.mj_forward(self.model, self.data)

        # 4. Extract end-effector translational Jacobian (3 x 7)
        ee_pos = self.data.xpos[self.ee_id]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, ee_pos, self.ee_id)
        J_arm = jacp[:, self.dof_indices]

        # 5. Eigendecomposition/SVD of J_arm to extract principal axes
        U, S, Vt = np.linalg.svd(J_arm)

        # Center c: (0, 0, 0) in velocity space
        c = np.zeros(3, dtype=np.float32)

        # Radii r: scaled by max motor velocity limit (0.2 rad/s)
        r = (0.2 * S).astype(np.float32)

        # Rotation R = U. Convert U matrix to unit quaternion q_e
        R = U
        # Quaternion conversion
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S_val = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S_val
            qx = (R[2, 1] - R[1, 2]) / S_val
            qy = (R[0, 2] - R[2, 0]) / S_val
            qz = (R[1, 0] - R[0, 1]) / S_val
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S_val = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S_val
            qx = 0.25 * S_val
            qy = (R[0, 1] + R[1, 0]) / S_val
            qz = (R[0, 2] + R[2, 0]) / S_val
        elif R[1, 1] > R[2, 2]:
            S_val = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S_val
            qx = (R[0, 1] + R[1, 0]) / S_val
            qy = 0.25 * S_val
            qz = (R[1, 2] + R[2, 1]) / S_val
        else:
            S_val = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S_val
            qx = (R[0, 2] + R[2, 0]) / S_val
            qy = (R[1, 2] + R[2, 1]) / S_val
            qz = 0.25 * S_val

        q_e = np.array([qw, qx, qy, qz], dtype=np.float32)
        V_vol = np.array([4.0 / 3.0 * np.pi * r[0] * r[1] * r[2]], dtype=np.float32)

        return (
            torch.from_numpy(c),
            torch.from_numpy(r),
            torch.from_numpy(q_e),
            torch.from_numpy(V_vol),
        )

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)

        # 1. Inject controller gains telemetry into the batch
        B_steps = self.num_steps
        kp = self.default_kp.unsqueeze(0).repeat(B_steps, 1)  # [T, 7]
        kd = self.default_kd.unsqueeze(0).repeat(B_steps, 1)  # [T, 7]

        if "observation.controller_gains" in batch:
            gains = batch["observation.controller_gains"]
            kp = gains[..., :7]
            kd = gains[..., 7:]

        batch["kp"] = kp
        batch["kd"] = kd

        # 2. Check if ellipsoid parameters are already pre-cached to bypass on-the-fly SVD
        if (
            self.use_tensor_cache
            and self._last_loaded_data is not None
            and "ellipsoid_center" in self._last_loaded_data
        ):
            frame_idx = int(self.frame_indices[idx])
            seq_steps = torch.arange(frame_idx, frame_idx + self.num_steps)
            max_frame_idx = self._last_loaded_data["pixels"].shape[0] - 1
            clamped_steps = torch.clamp(seq_steps, 0, max_frame_idx)

            batch["ellipsoid_center"] = self._last_loaded_data["ellipsoid_center"][
                clamped_steps
            ]
            batch["ellipsoid_radii"] = self._last_loaded_data["ellipsoid_radii"][
                clamped_steps
            ]
            batch["ellipsoid_quat"] = self._last_loaded_data["ellipsoid_quat"][
                clamped_steps
            ]
            batch["ellipsoid_volume"] = self._last_loaded_data["ellipsoid_volume"][
                clamped_steps
            ]
        elif "observation.state" in batch:
            # Fallback to lazy on-the-fly SVD calculations if cache is missing these keys
            states = batch["observation.state"]  # [T, 64] or [T, 32]
            c_list, r_list, qe_list, v_list = [], [], [], []
            for t in range(B_steps):
                c, r, q_e, V_vol = self.compute_ellipsoid(states[t])
                c_list.append(c)
                r_list.append(r)
                qe_list.append(q_e)
                v_list.append(V_vol)

            batch["ellipsoid_center"] = torch.stack(c_list, dim=0)  # [T, 3]
            batch["ellipsoid_radii"] = torch.stack(r_list, dim=0)  # [T, 3]
            batch["ellipsoid_quat"] = torch.stack(qe_list, dim=0)  # [T, 4]
            batch["ellipsoid_volume"] = torch.stack(v_list, dim=0)  # [T, 1]

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

        db_z = torch.as_tensor(successful_database["z"], device=z_t.device)
        db_q = torch.as_tensor(successful_database["q"], device=q_t.device)
        db_dq = torch.as_tensor(successful_database["dq"], device=dq_t.device)

        # Compute batched composite distance
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
