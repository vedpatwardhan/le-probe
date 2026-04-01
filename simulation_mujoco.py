import numpy as np
import mujoco
import rerun as rr
import zmq
import msgpack
import logging
import torch
import time
import os
from lerobot_manager import LeRobotManager
from gr1_config import COMPACT_WIRE_JOINTS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX

# Mink for IK
import mink

# Configuration
SCENE_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/gr1_gr00t/sim_assets/urdf/scene_gr1_pickup.xml"
DEVICE = "cpu"

import os
import datetime
import warnings
from pathlib import Path
from PIL import Image

# Suppress performance warnings from qpsolvers
warnings.filterwarnings("ignore", category=UserWarning, module="qpsolvers")

class GR1MuJoCoSimulation:
    def __init__(self, scene_path=SCENE_PATH):
        print(f"--- HARDENED SIM v2.1 (IK Loop Fixed) ---")
        print(f"--- Loading MuJoCo Scene: {scene_path} ---")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        
        # Session & Directory Management
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_log_dir = Path("/Users/vedpatwardhan/Desktop/cortex-os/temp_images") / self.session_id
        self.cam_names = ["world_top", "world_left", "world_right", "world_center", "world_wrist"]
        
        print(f"[LOG] Initializing Image Persistence: {self.base_log_dir}")
        for cam in self.cam_names:
            (self.base_log_dir / cam).mkdir(parents=True, exist_ok=True)
        self.frame_indices = {cam: 0 for cam in self.cam_names}
        
        # Renderer for multiple camera views
        self.renderer = mujoco.Renderer(self.model, height=480, width=480)
        
        # Protocol configuration
        self.wire_min = np.array(JOINT_LIMITS_MIN)
        self.wire_max = np.array(JOINT_LIMITS_MAX)
        
        # Load authorized joints
        self.allowed_names = set()
        for path in ["gr1_gr00t/teleop_joints.txt", "gr1_gr00t/ik_joints.txt"]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.allowed_names.update([l.strip().split("#")[0].strip() for l in f if l.strip()])
        
        # Mapping Protocol Names -> Joint / Qpos IDs
        self.protocol_joint_ids = []
        self.v_allowed_mask = np.zeros(32)
        print("\n[INIT] Proxy -> MuJoCo Mapping Diagnostics:")
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.protocol_joint_ids.append(j_id)
                if name in self.allowed_names:
                    self.v_allowed_mask[i] = 1.0
                    q_idx = self.model.jnt_qposadr[j_id]
                    print(f"  Proxy [{i:02}] -> Joint {j_id:02} ({name:<30}) | pos_idx: {q_idx}")
            except:
                self.protocol_joint_ids.append(-1)
                
        # Finger Coupling Graph (Proximal -> Distal)
        self.coupling_map = {}
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            if "proximal" in name.lower():
                prefix = name.split("_proximal")[0]
                for j in range(self.model.njnt):
                    j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    if j_name and prefix in j_name and j_name != name:
                        if i not in self.coupling_map: self.coupling_map[i] = []
                        self.coupling_map[i].append(self.model.jnt_qposadr[j])
        
        # LeRobot Manager
        self.recorder = LeRobotManager(repo_id="gr1_pickup_mujoco", fps=10)
        
        # IK Setup (Mink 1.1.0) with DOF Freezing for Authorization
        self.ee_link_name = "R_index_tip_link"
        self.configuration = mink.Configuration(self.model)
        
        # ... (rest of configuration same)
        
        # Identify DOFs to freeze (Strict Authorization)
        # Root (0-5: FROZEN), Waist (6-8), Left Side (9-38: FROZEN), Right Side (39-45)
        all_dofs = set(range(self.model.nv))
        authorized_dofs = {6, 7, 8, 39, 40, 41, 42, 43, 44, 45}
        frozen_dofs = list(all_dofs - authorized_dofs)
        print(f"[IK] Surgical Mapping: {self.model.nv} total | {len(authorized_dofs)} authorized | {len(frozen_dofs)} frozen")

        self.tasks = [
            mink.FrameTask(
                frame_name=self.ee_link_name,
                frame_type="body",
                position_cost=2.0,  # Prioritize Position
                orientation_cost=0.1,  # Allow orientation compromise
                lm_damping=0.01,
            ),
            mink.PostureTask(model=self.model, cost=1e-3),
            mink.DofFreezingTask(model=self.model, dof_indices=frozen_dofs)
        ]
        self.tasks[1].set_target(self.model.qpos0)
        
        # Pre-set robot height in model defaults
        root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_q_idx = self.model.jnt_qposadr[root_id]
        self.data.qpos[self.root_q_idx + 2] = 0.95
        
        # Internal state
        self.last_target_q = self.data.qpos.copy()
        self.active_joints_this_command = set()
        self.is_recording = False
        self.is_running = True

    def get_state_32(self):
        """Returns normalized 32-DOF protocol vector [-1.0, 1.0]."""
        state = np.full(32, np.nan, dtype=np.float32)
        for i, j_id in enumerate(self.protocol_joint_ids):
            # Strict authorization check to prevent UI auto-activation
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                qpos_idx = self.model.jnt_qposadr[j_id]
                val = self.data.qpos[qpos_idx]
                rng = max(1e-4, self.wire_max[i] - self.wire_min[i])
                norm = 2.0 * (val - self.wire_min[i]) / rng - 1.0
                state[i] = np.clip(norm, -1.1, 1.1)
        return state

    def solve_ik(self, pos, quat):
        """Production-Grade IK Solver with Persistent Updates."""
        self.configuration.update(self.data.qpos)
        rotation = mink.SO3(np.array(quat))
        self.tasks[0].set_target(mink.SE3.from_rotation_and_translation(rotation, pos))
        
        # Authorized: Waist (6-8) + Right Arm (39-45)
        auth_dofs = {6, 7, 8, 39, 40, 41, 42, 43, 44, 45}
        
        # Convergence Loop
        for i in range(500):
            solver = mink.solve_ik(self.configuration, self.tasks, dt=0.05, solver="osqp")
            
            # 1. Capture snapshot to avoid 'Copy Trap'
            q_ref = self.configuration.q.copy()
            
            # 2. Explicitly freeze unauthorized velocities (Double-Lock)
            full_vel = solver.copy()
            for d in range(len(full_vel)):
                if d not in auth_dofs:
                    full_vel[d] = 0.0
            
            # 3. Integrate and Push Update back to Mink
            mujoco.mj_integratePos(self.model, q_ref, full_vel, 0.05)
            self.configuration.update(q_ref)
            
            # 4. Success Check
            err = self.tasks[0].compute_error(self.configuration)
            if np.linalg.norm(err) < 0.01:
                print(f"[REACH] Converged in {i} steps (Error: {np.linalg.norm(err)*1000:.1f}mm)")
                break
                
        return self.configuration.q.copy()

    def reset_env(self):
        """Parity Reset: Smooth Glide + Environment Diversity."""
        print("[RESET] Randomizing Environment...")
        
        # 1. Randomize Cube Position & Rotation
        rx = np.random.uniform(0.35, 0.55)
        ry = np.random.uniform(-0.15, 0.15)
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_q_idx = self.model.jnt_qposadr[cube_id]
        
        # Position
        self.data.qpos[cube_q_idx : cube_q_idx+3] = [rx, ry, 0.82]
        # Random Z-rotation for the cube
        angle = np.random.uniform(0, 2*np.pi)
        self.data.qpos[cube_q_idx+3 : cube_q_idx+7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
        
        # 2. Determine Home Pose (Start with neutral qpos0)
        home_q = self.model.qpos0.copy()
        
        # CRITICAL: Persist the randomized cube into the home pose
        home_q[cube_q_idx : cube_q_idx+7] = self.data.qpos[cube_q_idx : cube_q_idx+7].copy()
        
        # Only override authorized joints to their protocol home + JITTER
        for i, j_id in enumerate(self.protocol_joint_ids):
            if j_id != -1 and self.v_allowed_mask[i] > 0.5:
                mid = (self.wire_max[i] + self.wire_min[i]) / 2.0
                jitter = np.random.normal(0, 0.05) 
                home_q[self.model.jnt_qposadr[j_id]] = mid + jitter
        
        # Set Root Height
        home_q[self.root_q_idx + 2] = 0.95
        
        # Jitter Waist Pitch
        try:
            wp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
            home_q[self.model.jnt_qposadr[wp_id]] += np.random.uniform(-0.1, 0.1)
        except: pass
        
        self.last_target_q = home_q.copy()
        self.dispatch_action(None, home_q)
        print(f"Env randomized. Cube at ({rx:.2f}, {ry:.2f})")

    def process_target_32(self, action_32):
        """Processes normalized inputs with Authorize Masking and Finger Coupling."""
        self.active_joints_this_command.clear()
        for i, val in enumerate(action_32):
            if not np.isnan(val) and self.v_allowed_mask[i] > 0 and self.protocol_joint_ids[i] != -1:
                self.active_joints_this_command.add(i)
                q_idx = self.model.jnt_qposadr[self.protocol_joint_ids[i]]
                rad = (val + 1.0) / 2.0 * (self.wire_max[i] - self.wire_min[i]) + self.wire_min[i]
                self.last_target_q[q_idx] = rad
                
                # Apply coupling (mimic)
                if i in self.coupling_map:
                    for distal_idx in self.coupling_map[i]:
                        self.last_target_q[distal_idx] = rad

    def print_joint_status(self):
        """Parity Diagnostic: Target vs Actual Table."""
        if not self.active_joints_this_command: return
        print("\n[OUTPUT] Action Finished. Target vs Actual:")
        for idx in sorted(list(self.active_joints_this_command)):
            name = COMPACT_WIRE_JOINTS[idx]
            q_idx = self.model.jnt_qposadr[self.protocol_joint_ids[idx]]
            tar = self.last_target_q[q_idx]
            act = self.data.qpos[q_idx]
            print(f"  {name:<30} | Tar: {tar:6.3f} | Act: {act:6.3f} | Err: {act-tar:6.3f}")
        print("-" * 50)

    def render_and_record(self, action_32):
        """Unified Rerun + LeRobot + Disk Persistence for 5 Cameras."""
        views = {}
        
        for name in self.cam_names:
            try:
                self.renderer.update_scene(self.data, camera=name)
                rgb = self.renderer.render()
                views[name] = rgb
                rr.log(name, rr.Image(rgb))
                
                # Serial Disk Serialization
                img_path = self.base_log_dir / name / f"{self.frame_indices[name]:04d}.jpg"
                Image.fromarray(rgb).save(img_path)
                self.frame_indices[name] += 1
            except Exception as e:
                pass # Camera might not be reachable yet
                
        if self.is_recording:
            self.recorder.add_frame(views, self.get_state_32(), action_32)

    def dispatch_action(self, action_32, target_q):
        """1.0s Fixed Window Interpolation."""
        num_steps = 100 # Smooth enough for visual
        start_q = self.data.qpos.copy()
        for i in range(num_steps):
            alpha = i / float(num_steps)
            self.data.qpos[:] = start_q + alpha * (target_q - start_q)
            mujoco.mj_fwdPosition(self.model, self.data)
            if i % 10 == 0:
                self.render_and_record(action_32)
                time.sleep(0.005) # Force visual separation
        
        self.data.qpos[:] = target_q
        mujoco.mj_fwdPosition(self.model, self.data)
        self.print_joint_status()

    def run(self, port=5556):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")
        
        rr.init("gr1_teleop_mujoco", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        print(f"🚀 Hardened MuJoCo Sim Running on port {port} (Parity Level: 100%)")
        
        while self.is_running:
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            cmd = data.get("command")
            
            # Universal metadata response helper
            def send_resp(payload):
                payload.update({
                    "upload_queue": self.recorder.pending_uploads,
                    "total_episodes": self.recorder.total_episodes
                })
                socket.send(msgpack.packb(payload))

            if cmd == "reset":
                self.reset_env()
                send_resp({"status": "reset_ok", "joints": self.get_state_32().tolist()})
            elif cmd == "start_recording":
                self.recorder.start_episode(data.get("task", "Pick up red cube"))
                self.is_recording = True
                send_resp({"status": "recording_started"})
            elif cmd == "stop_recording":
                self.recorder.stop_episode()
                self.is_recording = False
                send_resp({"status": "recording_stopped"})
            elif cmd == "auto_reach":
                cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
                cube_pos = self.data.qpos[self.model.jnt_qposadr[cube_id] : self.model.jnt_qposadr[cube_id]+3].copy()

                # Stable top-down orientation: [0, 1, 0, 0] (wxyz)
                target_q = self.solve_ik(cube_pos, [0, 1, 0, 0])
                
                print(f"🎯 Auto-Reach: Target {target_pos} | Quat [0,1,0,0]")
                self.dispatch_action(None, target_q)
                send_resp({"status": "auto_reach_ok", "joints": self.get_state_32().tolist()})
            elif cmd == "poll_status":
                send_resp({"status": "status_ok"})
            elif "target" in data:
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)
                self.dispatch_action(action_32, self.last_target_q)
                send_resp({"status": "step_ok"})
            else:
                send_resp({"status": "unknown"})

if __name__ == "__main__":
    GR1MuJoCoSimulation().run()
