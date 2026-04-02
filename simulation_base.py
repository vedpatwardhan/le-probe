import numpy as np
import mujoco
import rerun as rr
import os
import datetime
import warnings
from pathlib import Path
from PIL import Image
import mink

from lerobot_manager import LeRobotManager
from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    SCENE_PATH,
)

# Suppress performance warnings from qpsolvers
warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")

class GR1MuJoCoBase:
    """
    Shared Physical Foundation for GR-1 MuJoCo Simulations.
    Handles XML loading, IK solving, State extraction, and Perception.
    """
    def __init__(self, scene_path=SCENE_PATH):
        print(f"--- GR-1 MODULAR BASE (MuJoCo) ---")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        
        # Session & Directory Management
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = Path("/Users/vedpatwardhan/Desktop/cortex-os/temp_images") / self.session_id
        self.cam_names = ["world_top", "world_left", "world_right", "world_center", "world_wrist"]
        
        for cam in self.cam_names:
            (self.base_log_dir / cam).mkdir(parents=True, exist_ok=True)
        self.frame_indices = {cam: 0 for cam in self.cam_names}

        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=480, width=480)

        # Mapping Protocol Names -> Joint IDs
        self.wire_min = np.array(JOINT_LIMITS_MIN)
        self.wire_max = np.array(JOINT_LIMITS_MAX)
        self._init_joint_mappings()
        self._init_finger_coupling()
        
        # LeRobot Manager
        self.recorder = LeRobotManager(repo_id="gr1_pickup_mujoco", fps=10)

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

    def _init_joint_mappings(self):
        self.allowed_names = set()
        for path in ["gr1_gr00t/teleop_joints.txt", "gr1_gr00t/ik_joints.txt"]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.allowed_names.update([l.strip().split("#")[0].strip() for l in f if l.strip()])

        self.protocol_joint_ids = []
        self.v_allowed_mask = np.zeros(32)
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.protocol_joint_ids.append(j_id)
                if name in self.allowed_names:
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
                        if "thumb" in name.lower() and (("yaw" in name.lower() and "pitch" in j_name.lower()) or ("pitch" in name.lower() and "yaw" in j_name.lower())):
                            continue
                        if i not in self.coupling_map: self.coupling_map[i] = []
                        self.coupling_map[i].append(self.model.jnt_qposadr[j])

    def _init_ik_solver(self):
        self.ee_index_link = "R_index_tip_link"
        self.ee_thumb_link = "R_thumb_tip_link"
        self.ee_wrist_link = "right_hand_pitch_link"
        self.configuration = mink.Configuration(self.model)
        all_dofs = set(range(self.model.nv))
        self.auth_dofs = {18, 19, 20, 39, 40, 41, 42, 43, 44, 45}
        frozen_dofs = list(all_dofs - self.auth_dofs)
        self.tasks = [
            mink.FrameTask(frame_name=self.ee_index_link, frame_type="body", position_cost=5.0, orientation_cost=0.0, lm_damping=0.001),
            mink.FrameTask(frame_name=self.ee_thumb_link, frame_type="body", position_cost=5.0, orientation_cost=0.0, lm_damping=0.001),
            mink.FrameTask(frame_name=self.ee_wrist_link, frame_type="body", position_cost=2.0, orientation_cost=0.0, lm_damping=0.001),
            mink.PostureTask(model=self.model, cost=1e-6),
            mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs),
        ]
        self.tasks[3].set_target(self.model.qpos0)

    def get_state_32(self):
        state = np.full(32, np.nan, dtype=np.float32)
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                qpos_idx = self.model.jnt_qposadr[j_id]
                val = self.data.qpos[qpos_idx]
                rng = max(1e-4, self.wire_max[i] - self.wire_min[i])
                norm = 2.0 * (val - self.wire_min[i]) / rng - 1.0
                state[i] = np.clip(norm, -1.1, 1.1)
        return state

    def solve_ik(self, pos_wrist, quat, pos_index=None, pos_thumb=None):
        if pos_index is None: pos_index = pos_wrist + [0, 0, 0.05]
        if pos_thumb is None: pos_thumb = pos_wrist + [0, 0, 0.05]
        self.configuration.update(self.data.qpos)
        rotation = mink.SO3(np.array(quat))
        self.tasks[0].set_target(mink.SE3.from_rotation_and_translation(rotation, pos_index))
        self.tasks[1].set_target(mink.SE3.from_rotation_and_translation(rotation, pos_thumb))
        self.tasks[2].set_target(mink.SE3.from_rotation_and_translation(rotation, pos_wrist))
        for i in range(1500):
            solver = mink.solve_ik(self.configuration, self.tasks, dt=0.15, solver="osqp")
            q_ref = self.configuration.q.copy()
            full_vel = solver.copy()
            for d in range(len(full_vel)):
                if d not in self.auth_dofs: full_vel[d] = 0.0
            mujoco.mj_integratePos(self.model, q_ref, full_vel, 0.05)
            self.configuration.update(q_ref)
            err = self.tasks[0].compute_error(self.configuration)
            if np.linalg.norm(err) < 0.01: break
        return self.configuration.q.copy()

    def sync_ctrl_to_qpos(self, q):
        for a_id in range(self.model.nu):
            j_id = self.model.actuator_trnid[a_id, 0]
            q_idx = self.model.jnt_qposadr[j_id]
            self.data.ctrl[a_id] = q[q_idx]

    def render_and_record(self, action_32):
        views = {}
        for name in self.cam_names:
            self.renderer.update_scene(self.data, camera=name)
            rgb = self.renderer.render()
            views[name] = rgb
            if self.rerun_count % 2 == 0: rr.log(name, rr.Image(rgb))
            img_path = self.base_log_dir / name / f"{self.frame_indices[name]:04d}.jpg"
            Image.fromarray(rgb).save(img_path)
            self.frame_indices[name] += 1
        self.rerun_count += 1
        if self.is_recording: self.recorder.add_frame(views, self.get_state_32(), action_32)

    def dispatch_action(self, action_32, target_q):
        total_steps = 200
        start_q = self.data.qpos.copy()
        root_target = target_q[self.root_q_idx : self.root_q_idx + 7]
        for step in range(total_steps):
            alpha = (step + 1) / float(total_steps)
            current_target_q = start_q + alpha * (target_q - start_q)
            self.sync_ctrl_to_qpos(current_target_q)
            self.data.qpos[self.root_q_idx : self.root_q_idx + 7] = root_target
            self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)
            if step % 16 == 0: self.render_and_record(action_32)

    def reset_env(self):
        print("🎲 Randomizing the environment...")
        rx, ry = np.random.uniform(0.35, 0.55), np.random.uniform(-0.15, 0.15)
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        self.data.qpos[cube_q_idx : cube_q_idx + 3] = [rx, ry, 0.82]
        angle = np.random.uniform(0, 2 * np.pi)
        self.data.qpos[cube_q_idx + 3 : cube_q_idx + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        home_q = self.model.qpos0.copy()
        home_q[cube_q_idx : cube_q_idx + 7] = self.data.qpos[cube_q_idx : cube_q_idx + 7].copy()
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                home_q[self.model.jnt_qposadr[j_id]] = (self.wire_max[i] + self.wire_min[i]) / 2.0 + np.random.normal(0, 0.05)
        home_q[self.root_q_idx : self.root_q_idx + 3] = [0.0, 0.0, 0.95]
        self.last_target_q = home_q.copy()
        self.dispatch_action(None, home_q)

    def process_target_32(self, action_32):
        self.active_joints_this_command.clear()
        for i, val in enumerate(action_32):
            if (
                not np.isnan(val)
                and self.v_allowed_mask[i] > 0
                and self.protocol_joint_ids[i] != -1
            ):
                self.active_joints_this_command.add(i)
                val = np.clip(val, -1.0, 1.0)
                q_idx = self.model.jnt_qposadr[self.protocol_joint_ids[i]]
                rad = (
                    (val + 1.0) / 2.0 * (self.wire_max[i] - self.wire_min[i])
                    + self.wire_min[i]
                )
                self.last_target_q[q_idx] = rad
                if i in self.coupling_map:
                    for distal_idx in self.coupling_map[i]:
                        self.last_target_q[distal_idx] = rad
