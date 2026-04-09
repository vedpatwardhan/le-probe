import numpy as np
import mujoco
import rerun as rr
import os
import datetime
import warnings
from pathlib import Path
from PIL import Image
import mink

from research.lerobot_manager import LeRobotManager
from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    SCENE_PATH,
)

# Suppress performance warnings from qpsolvers
warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")

try:
    import pycapacity.robot as pycap
    PYCAPACITY_AVAILABLE = True
except ImportError:
    PYCAPACITY_AVAILABLE = False


class GR1MuJoCoBase:
    """
    Shared Physical Foundation for GR-1 MuJoCo Simulations.
    Handles XML loading, IK solving, State extraction, and Perception.
    """

    def __init__(self, scene_path=SCENE_PATH, restrict_ik=True):
        print(f"--- GR-1 MODULAR BASE (MuJoCo) ---")
        self.restrict_ik = restrict_ik
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Perception & Cameras
        self.cam_names = [
            "world_top",
            "world_left",
            "world_right",
            "world_center",
            "world_wrist",
        ]
        self.frame_indices = {cam: 0 for cam in self.cam_names}

        # Renderer
        self.res = (480, 480)
        self.renderer = mujoco.Renderer(
            self.model, height=self.res[1], width=self.res[0]
        )

        # Mapping Protocol Names -> Joint IDs
        self.wire_min = np.array(JOINT_LIMITS_MIN)
        self.wire_max = np.array(JOINT_LIMITS_MAX)
        self._init_joint_mappings()
        self._init_finger_coupling()

        # Diagnostic Logging
        self.debug_log_path = None

        # LeRobot Manager
        self.recorder = LeRobotManager(
            repo_id="vedpatwardhan/gr1_pickup_large", fps=10, upload_interval=20
        )

        # IK Setup (Mink 1.1.0)
        self._init_ik_solver()

        # Internal state
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_q_idx = self.model.jnt_qposadr[root_id]
        self.data.qpos[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]

        self.last_target_q = self.data.qpos.copy()
        self.active_joints_this_command = set()
        self.is_recording = False
        self.rerun_count = 0
        self.render_step_idx = 0
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "temp_images"
            / self.session_id
        )
        self._init_joint_mappings()
        self._init_finger_coupling()

    def _init_joint_mappings(self):
        self.ik_names = set()
        base_path = os.path.dirname(os.path.abspath(__file__))

        # Load IK whitelist
        i_path = os.path.join(base_path, "ik_joints.txt")
        with open(i_path, "r") as f:
            self.ik_names.update(
                [l.strip().split("#")[0].strip() for l in f if l.strip()]
            )

        print(f"✅ Loaded {len(self.ik_names)} IK joint names.")

        self.protocol_joint_ids = []
        self.v_allowed_mask = np.zeros(32)
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.protocol_joint_ids.append(j_id)
                # v_allowed_mask is ALWAYS 1.0 for valid joints to allow full protocol freedom
                # (randomization, VLA actions, etc.), even if IK is restricted.
                if j_id != -1:
                    self.v_allowed_mask[i] = 1.0
            except:
                self.protocol_joint_ids.append(-1)

    def _init_finger_coupling(self):
        self.coupling_map = {}
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            if "proximal" in name.lower():
                base_prefix = name.split("_proximal")[0]
                for j in range(self.model.njnt):
                    j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if j_name and base_prefix in j_name and j_name != name:
                        if "thumb" in name.lower() and (
                            ("yaw" in name.lower() and "pitch" in j_name.lower())
                            or ("pitch" in name.lower() and "yaw" in j_name.lower())
                        ):
                            continue
                        if i not in self.coupling_map:
                            self.coupling_map[i] = []
                        # Store qpos index instead of joint id!
                        q_idx = self.model.jnt_qposadr[j]
                        self.coupling_map[i].append(q_idx)

    def _init_ik_solver(self):
        self.ee_index_link = "R_index_tip_link"
        self.ee_thumb_link = "R_thumb_tip_link"
        self.ee_wrist_link = "right_hand_pitch_link"
        self.configuration = mink.Configuration(self.model)

        # Determine authorized velocity indices (DOFs) for the IK Solver
        self.auth_dofs = set()
        whitelist = self.ik_names if self.restrict_ik else COMPACT_WIRE_JOINTS
        for name in whitelist:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                v_idx = self.model.jnt_dofadr[j_id]
                # Map joint type (free=0, ball=1, slide=2, hinge=3) to DOF count
                dof_counts = [6, 3, 1, 1]
                j_type = self.model.jnt_type[j_id]
                dnum = dof_counts[j_type]
                for d in range(dnum):
                    self.auth_dofs.add(int(v_idx + d))
        print(f"IK auth_dofs (restricted={self.restrict_ik}):", self.auth_dofs)

        # Always allow root for relative calculations if needed,
        # but here we strictly follow the whitelist for robot bones.

        all_dofs = set(range(self.model.nv))
        frozen_dofs = list(all_dofs - self.auth_dofs)

        self.tasks = [
            mink.FrameTask(
                frame_name=self.ee_index_link,
                frame_type="body",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.FrameTask(
                frame_name=self.ee_thumb_link,
                frame_type="body",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.FrameTask(
                frame_name=self.ee_wrist_link,
                frame_type="body",
                position_cost=2.0,
                orientation_cost=0.0,
                lm_damping=0.001,
            ),
            mink.PostureTask(model=self.model, cost=1e-6),
            mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs),
        ]
        self.tasks[3].set_target(self.model.qpos0)

    def _debug_log(self, msg):
        """Helper for timestamped diagnostic logging. Only writes if debug_log_path is set."""
        if self.debug_log_path is None:
            return
        t_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(self.debug_log_path, "a") as f:
            f.write(f"[{t_str}] {msg}\n")

    def _post_render_hook(self, name, rgb):
        """Saves camera views to the filesystem for diagnostic-verification (Mirrors original teleop logic)."""
        # Save in a subdirectory for this specific camera inside the session folder
        cam_dir = self.base_log_dir / name
        cam_dir.mkdir(parents=True, exist_ok=True)

        img_path = cam_dir / f"{self.frame_indices[name]:04d}.png"
        Image.fromarray(rgb).save(img_path)

    def get_reachability_metrics(self):
        """Calculates the velocity polytope volume for the hands using pycapacity."""
        if not PYCAPACITY_AVAILABLE:
            return {}

        metrics = {}
        for side in ["left", "right"]:
            ee_name = f"{side}_hand_pitch_link"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
            if body_id == -1:
                continue

            # 1. Full Jacobian
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac, None, body_id)

            # 2. Joint Limits (Normalizing velocity to 1.0 rad/s for relative volume)
            dq_max = np.ones(self.model.nv)
            dq_min = -np.ones(self.model.nv)

            try:
                # Calculate the 3D velocity polytope
                poly = pycap.velocity_polytope(jac, dq_min, dq_max)
                metrics[f"reachability/{side}_hand_volume"] = float(poly.volume)
            except Exception as e:
                # Handle singular configurations where polytope might fail
                metrics[f"reachability/{side}_hand_volume"] = 0.0

        return metrics

    def get_state_32(self):
        return self.qpos_to_action_32(self.data.qpos)

    def qpos_to_action_32(self, qpos):
        """Converts a 76-dim qpos vector to a normalized 32-dim protocol action/state."""
        state = np.zeros(32, dtype=np.float32)
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1:
                qpos_idx = self.model.jnt_qposadr[j_id]
                val = qpos[qpos_idx]
                rng = max(1e-4, self.wire_max[i] - self.wire_min[i])
                norm = 2.0 * (val - self.wire_min[i]) / rng - 1.0
                state[i] = np.clip(norm, -1.1, 1.1)
        return state

    def solve_ik(self, pos_wrist, quat, pos_index=None, pos_thumb=None):
        """Hardened IK solver for a 3-tip end effector (Index, Thumb, Wrist)."""
        quat = np.array(quat)
        if quat.shape[0] != 4:
            raise ValueError(f"solve_ik: quat must be length 4 (wxyz), got {len(quat)}")

        pos_wrist = np.array(pos_wrist)
        if pos_index is None:
            pos_index = pos_wrist + np.array([0, 0, 0.05])
        if pos_thumb is None:
            pos_thumb = pos_wrist + np.array([0, 0, 0.05])

        pos_index = np.array(pos_index)
        pos_thumb = np.array(pos_thumb)

        self.configuration.update(self.data.qpos)
        rotation = mink.SO3(quat)
        self.tasks[0].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_index)
        )
        self.tasks[1].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_thumb)
        )
        self.tasks[2].set_target(
            mink.SE3.from_rotation_and_translation(rotation, pos_wrist)
        )
        for i in range(1500):
            solver = mink.solve_ik(
                self.configuration, self.tasks, dt=0.15, solver="osqp"
            )
            q_ref = self.configuration.q.copy()
            full_vel = solver.copy()
            for d in range(len(full_vel)):
                if d not in self.auth_dofs:
                    full_vel[d] = 0.0
            mujoco.mj_integratePos(self.model, q_ref, full_vel, 0.05)
            self.configuration.update(q_ref)
            err = self.tasks[0].compute_error(self.configuration)
            if np.linalg.norm(err) < 0.01:
                break
        return self.configuration.q.copy()

    def sync_ctrl_to_qpos(self, q):
        for a_id in range(self.model.nu):
            j_id = self.model.actuator_trnid[a_id, 0]
            q_idx = self.model.jnt_qposadr[j_id]
            self.data.ctrl[a_id] = q[q_idx]

    def render_and_record(self, action_32):
        views = {}
        rr.set_time("sim_step", sequence=self.render_step_idx)

        for name in self.cam_names:
            self.renderer.update_scene(self.data, camera=name)
            rgb = self.renderer.render()
            views[name] = rgb
            if self.rerun_count % 2 == 0:
                rr.log(name, rr.Image(rgb))

            self._post_render_hook(name, rgb)
            self.frame_indices[name] += 1

        self.render_step_idx += 1
        self.rerun_count += 1

        # Periodic Reachability Logging & Dataset Grounding
        reach_metrics = None
        if (self.is_recording or self.rerun_count % 10 == 0) and PYCAPACITY_AVAILABLE:
            reach_metrics = self.get_reachability_metrics()

        # Throttled Rerun Logging (1Hz)
        if self.rerun_count % 10 == 0 and reach_metrics:
            for k, v in reach_metrics.items():
                rr.log(f"telemetry/{k}", rr.Scalar(v))

        if self.is_recording:
            # Grounding: Save reachability alongside kinematic state
            self.recorder.add_frame(views, self.get_state_32(), action_32, reachability=reach_metrics)

    def dispatch_action(self, action_32, target_q):
        total_steps = 200
        start_q = self.data.qpos.copy()
        delta_norm = np.linalg.norm(target_q - start_q)
        self._debug_log(f"🚀 Dispatching Action. L2 Delta Norm: {delta_norm:.6f}")

        root_target = target_q[self.root_q_idx : self.root_q_idx + 7]
        for step in range(total_steps):
            alpha = (step + 1) / float(total_steps)
            current_target_q = start_q + alpha * (target_q - start_q)
            self.sync_ctrl_to_qpos(current_target_q)
            self.data.qpos[self.root_q_idx : self.root_q_idx + 7] = root_target
            self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)
            if step % 16 == 0:
                self.render_and_record(action_32)

    def reset_env(self):
        print("🎲 Randomizing the environment...")
        # Constrain cube randomization to table bounds (X: 0.25-0.65, Y: ±0.25)
        # Using [0.27, 0.63] and ±0.23 for a small safety margin from the edges
        rx, ry = np.random.uniform(0.27, 0.63), np.random.uniform(-0.23, 0.23)
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        self.data.qpos[cube_q_idx : cube_q_idx + 3] = [rx, ry, 0.82]
        angle = np.random.uniform(0, 2 * np.pi)
        self.data.qpos[cube_q_idx + 3 : cube_q_idx + 7] = [
            np.cos(angle / 2),
            0,
            0,
            np.sin(angle / 2),
        ]
        home_q = self.model.qpos0.copy()
        home_q[cube_q_idx : cube_q_idx + 7] = self.data.qpos[
            cube_q_idx : cube_q_idx + 7
        ].copy()
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                # Wider joint randomization range (±0.2 rad) to test starting configuration limits
                home_q[self.model.jnt_qposadr[j_id]] = (
                    self.wire_max[i] + self.wire_min[i]
                ) / 2.0 + np.random.uniform(-0.2, 0.2)
        home_q[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]
        self.last_target_q = home_q.copy()
        self.dispatch_action(None, home_q)

    def process_target_32(self, action_32):
        self.active_joints_this_command.clear()
        valid_actions = action_32[~np.isnan(action_32)]
        if len(valid_actions) > 0:
            self._debug_log(
                f"📥 process_target_32: stats [min:{np.min(valid_actions):.3f}, max:{np.max(valid_actions):.3f}, mean:{np.mean(valid_actions):.3f}]"
            )
        else:
            self._debug_log("⚠️ process_target_32: action_32 is ALL NAN")
        for i, val in enumerate(action_32):
            if (
                not np.isnan(val)
                and self.v_allowed_mask[i] > 0
                and self.protocol_joint_ids[i] != -1
            ):
                self.active_joints_this_command.add(i)
                val = np.clip(val, -1.0, 1.0)
                q_idx = self.model.jnt_qposadr[self.protocol_joint_ids[i]]
                rad = (val + 1.0) / 2.0 * (
                    self.wire_max[i] - self.wire_min[i]
                ) + self.wire_min[i]
                old_rad = self.last_target_q[q_idx]
                self.last_target_q[q_idx] = rad
                if abs(rad - old_rad) > 1e-5:
                    self._debug_log(
                        f"   🔗 Joint {i} ({COMPACT_WIRE_JOINTS[i]}): {old_rad:.4f} -> {rad:.4f}"
                    )
                if i in self.coupling_map:
                    for distal_idx in self.coupling_map[i]:
                        self.last_target_q[distal_idx] = rad
