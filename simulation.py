import numpy as np
import genesis as gs
import rerun as rr
import zmq
import msgpack
import logging
import torch
from lerobot_manager import LeRobotManager
from gr1_config import COMPACT_WIRE_JOINTS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX

URDF_PATH = "./repos/Wiki-GRx-Models/GRX/GR1/gr1t2/urdf/gr1t2_fourier_hand_6dof.urdf"

# -----------------------------------------------------------------------------
# CONFIG & LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(message)s")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)


class GR1Simulation:
    def __init__(self, urdf_path=URDF_PATH, test_mode=False):
        self.test_mode = test_mode
        # Initialize Genesis with re-init guard
        try:
            gs.init(backend=gs.gpu)
        except Exception:
             # Genesis likely already initialized in this process (e.g. during pytest)
             pass

        # Create Scene
        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.002, substeps=1),
            vis_options=gs.options.VisOptions(
                lights=[
                    {
                        "type": "directional",
                        "dir": (0, 0, -1),
                        "color": (1.0, 1.0, 1.0),
                        "intensity": 2.0,
                    }
                ]
            ),
            renderer=gs.renderers.Rasterizer(),
        )

        # Add entities
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, pos=(-0.2, 0, 0.95), fixed=True)
        )
        self.table = self.scene.add_entity(
            gs.morphs.Box(pos=(0.45, 0, 0.4), size=(0.4, 0.5, 0.8), fixed=True)
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(pos=(0.45, -0.2, 0.82), size=(0.04, 0.04, 0.04), fixed=False),
            surface=gs.surfaces.Default(color=(1, 0, 0)),
        )

        # Add cameras for top, left, right, center and wrist views
        self.world_cam_top = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(0.3, 0, 1.8), lookat=(0.3, 0, 0.8)
        )
        self.world_cam_left = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(0.45, -0.8, 1.1), lookat=(0.45, 0, 0.8)
        )
        self.world_cam_right = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(0.45, 0.8, 1.1), lookat=(0.45, 0, 0.8)
        )
        self.world_cam_center = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.2, 0, 1.1), lookat=(0.45, 0, 0.8)
        )
        self.world_cam_wrist = self.scene.add_camera(
            res=(224, 224),
            fov=110,
            pos=(0, 0, 0.05),
            lookat=(0, 0, 1.0),
            up=(0, -1, 0),
        )

        # Build the scene
        self.scene.build()

        # Attach wrist camera after build with a transformation matrix
        pos, lookat, up = (
            np.array([0, 0, 0.05]),
            np.array([0, 0, 1.0]),
            np.array([0, -1, 0]),
        )
        z = (lookat - pos) / np.linalg.norm(lookat - pos)
        x = np.cross(up, z) / np.linalg.norm(np.cross(up, z))
        y = np.cross(z, x)
        offset_T = np.eye(4)
        offset_T[:3, :] = np.column_stack([x, y, z, pos])
        self.world_cam_wrist.attach(
            rigid_link=self.robot.get_link("right_hand_pitch_link"),
            offset_T=offset_T,
        )

        # Gains & Control (chosen after a lot of trial-and-error)
        kp_array = np.full(self.robot.n_dofs, 3500.0)
        kv_array = np.full(self.robot.n_dofs, 150.0)
        self.robot.set_dofs_kp(kp_array)
        self.robot.set_dofs_kv(kv_array)

        # Force Range (chosen after a lot of trial-and-error)
        force_max = np.full(self.robot.n_dofs, 500.0)
        force_min = -force_max
        self.robot.set_dofs_force_range(force_min, force_max)

        # Internal state
        self.target_buffer = np.full(32, np.nan, dtype=np.float32)
        self.last_target_q = np.zeros(self.robot.n_dofs, dtype=np.float32)
        self.is_running = True
        self.active_joints_this_command = set()

        # LeRobot Recording State (target 10Hz for dataset)
        self.recorder = LeRobotManager(repo_id="gr1_pickup_large", fps=10)
        self.is_recording = False
        self.current_task = "Pick up the red cube"

        # Pre-compute protocol limit tensors on the correct device
        self.wire_min = torch.tensor(
            JOINT_LIMITS_MIN, device=DEVICE, dtype=torch.float32
        )
        self.wire_max = torch.tensor(
            JOINT_LIMITS_MAX, device=DEVICE, dtype=torch.float32
        )

        # Precomputed mapping from joint to dof
        self.joint_dof_map = []
        for idx, joint_name in enumerate(COMPACT_WIRE_JOINTS):
            joint = self.robot.get_joint(joint_name)
            dof_idx = joint.dofs_idx[0]
            limit_min, limit_max = JOINT_LIMITS_MIN[idx], JOINT_LIMITS_MAX[idx]

            # Finger coupling
            coupled_dofs = []
            if "proximal" in joint_name.lower():
                finger_prefix = joint_name.split("_proximal")[0]
                for other_joint in self.robot.joints:
                    if other_joint.name and finger_prefix in other_joint.name:
                        if (
                            other_joint.name != joint_name
                            and "proximal" not in other_joint.name.lower()
                        ):
                            coupled_dofs.append(other_joint.dofs_idx[0])

            self.joint_dof_map.append(
                {
                    "dof_idx": dof_idx,
                    "limits": (limit_min, limit_max),
                    "name": joint_name,
                    "coupled": coupled_dofs,
                }
            )

        # Save the natural URDF Home Pose to prevent "joint leakage" during resets
        self.home_q = self.robot.get_qpos().clone()

        # Inverse Kinematics Setup
        self.ee_link = self.robot.get_link("right_hand_roll_link")

        # Combine IK and Teleop into a 32-DOF reporting whitelist
        self.allowed_32_indices = set()
        self.allowed_names_set = set()

        # Load IK joints
        with open("gr1_gr00t/ik_joints.txt", "r") as f:
            self.allowed_names_set.update([l.strip() for l in f if l.strip()])

        # Load teleop joints
        with open("gr1_gr00t/teleop_joints.txt", "r") as f:
            self.allowed_names_set.update([l.strip() for l in f if l.strip()])

        # Map these names to the 32-DOF indices in our wire protocol
        for i, name in enumerate(COMPACT_WIRE_JOINTS):
            if name in self.allowed_names_set:
                self.allowed_32_indices.add(i)

        # Also maintain raw DOF indices for IK solver jitter
        self.ik_dof_indices = [
            j.dofs_idx[0] for j in self.robot.joints if j.name in self.allowed_names_set
        ]

        print(
            f"✅ Unified Mask initialized with {len(self.allowed_32_indices)} authorized joints."
        )
        for i in sorted(self.allowed_32_indices):
            mapping = self.joint_dof_map[i]
            print(
                f"  - Authorized Proxy [{i:02}] -> Robot DOF {mapping['dof_idx']} ({mapping['name']})"
            )

    def solve_ik(self, pos, quat):
        """Solves IK for the right arm, only updating the masked joints."""
        # Get proposed full-body IK from Genesis
        q_proposed = self.robot.inverse_kinematics(
            link=self.ee_link, pos=pos, quat=quat
        )

        # Overlay the masked joints onto our current pose
        q_final = self.robot.get_qpos().clone()
        for idx in self.ik_dof_indices:
            q_final[idx] = q_proposed[idx]

        # Override only our authorized joints and clamp them directly
        for i, mapping in enumerate(self.joint_dof_map):
            if i in self.allowed_32_indices:
                dof_idx = mapping["dof_idx"]
                # Apply the proposed IK value and immediately clamp it by our protocol limits
                q_final[dof_idx] = torch.clamp(
                    q_proposed[dof_idx], self.wire_min[i], self.wire_max[i]
                )

        return q_final

    def get_state_32(self):
        """Returns the current 32-DOF joint positions as a NumPy array for I/O."""
        raw_full_state = self.robot.get_dofs_position()
        # Extract only the 32 joints, keeping everything on device for now
        raw_state_32 = torch.stack(
            [raw_full_state[m["dof_idx"]] for m in self.joint_dof_map]
        )
        # Normalize in torch space, then move to CPU for the dataset/UI
        return self._normalize_state(raw_state_32).cpu().numpy()

    def _normalize_state(self, raw_state):
        """Normalizes raw joint positions to [-1, 1] using protocol constants."""
        # Ensure we are working with torch tensors for batch math
        if not isinstance(raw_state, torch.Tensor):
            raw_state = torch.tensor(raw_state, device=DEVICE, dtype=torch.float32)

        joint_range = self.wire_max - self.wire_min
        safe_range = torch.where(joint_range > 1e-4, joint_range, 1e-4)

        normalized_state = (raw_state - self.wire_min) / safe_range * 2.0 - 1.0

        # Create a mask for unauthorized joints (on device)
        mask = torch.zeros_like(normalized_state)
        for idx in self.allowed_32_indices:
            mask[idx] = 1.0

        normalized_state *= mask
        return torch.clamp(normalized_state, -1.0, 1.0)

    def reset_env(self):
        """Randomizes cube and robot arm with a safe backward lean."""
        # Cube randomization
        rx = np.random.uniform(0.35, 0.55)
        ry = np.random.uniform(-0.15, 0.15)
        self.cube.set_pos((rx, ry, 0.82))

        # Start from the current live pose to ensure unauthorized joints never budge
        q = self.robot.get_qpos().clone()

        # Jitter only authorized IK joints by ±0.2 rad
        jitter = (torch.rand(len(self.ik_dof_indices), device=DEVICE) - 0.5) * 0.4
        for i, dof_idx in enumerate(self.ik_dof_indices):
            q[dof_idx] = self.home_q[dof_idx] + jitter[i]

        # Force Waist Pitch to be forward/ready (Source of Truth)
        waist_pitch_joint = self.robot.get_joint("waist_pitch_joint")
        q[waist_pitch_joint.dofs_idx[0]] = np.random.uniform(-0.17, 0.18)

        # Diagnostic Logging
        print(f"\n[RESET] Randomization Trace:")
        curr_q = q.cpu().numpy()
        home_q = self.home_q.cpu().numpy()
        changed_count = 0
        for i in range(len(curr_q)):
            if abs(curr_q[i] - home_q[i]) > 1e-5:
                # Find joint name for this DOF
                name = "unknown"
                for j in self.robot.joints:
                    if j.dofs_idx and j.dofs_idx[0] == i:
                        name = j.name
                        break
                print(
                    f"  DOF {i:02} ({name:<30}) changed: {home_q[i]:.4f} -> {curr_q[i]:.4f}"
                )
                changed_count += 1
        print(f"Total Modified DOFs: {changed_count}")

        # Smooth Glide to Randomized Pose instead of teleporting
        self.dispatch_action(
            200, 20, np.full(32, np.nan), q, start_q=self.robot.get_qpos().clone()
        )

        self.last_target_q = q.clone()
        print(
            f"[RESET] Env randomized. Cube: ({rx:.2f}, {ry:.2f}) | "
            f"Waist Pitch: {q[waist_pitch_joint.dofs_idx[0]]:.2f}"
        )

    def process_target_32(self, action_32):
        """Maps 32-DOF actions to targets and logs progress."""
        any_update = False
        print(f"\n[INPUT] Received ZMQ message (32 DOFs):")
        for idx, mapping in enumerate(self.joint_dof_map):
            if idx not in self.allowed_32_indices:
                print(f"  [GATE] Skipping unauthorized index {idx} ({mapping['name']})")
                continue

            print(f"  [GATE] Allowing index {idx} ({mapping['name']})")
            val = action_32[idx]
            if np.isnan(val):
                continue

            self.active_joints_this_command.add(idx)
            val = np.clip(val, -1.0, 1.0)
            limit_min, limit_max = mapping["limits"]
            target_rad = (val + 1.0) / 2.0 * (limit_max - limit_min) + limit_min

            dof_idx = mapping["dof_idx"]
            print(
                f"  [{idx:02}] {mapping['name']:<30} | In: {val:6.3f} -> Tar: {target_rad:6.3f} rad"
            )

            if abs(self.last_target_q[dof_idx] - target_rad) > 1e-4:
                self.last_target_q[dof_idx] = float(target_rad)
                for c_idx in mapping["coupled"]:
                    self.last_target_q[c_idx] = float(target_rad)
                any_update = True
        return any_update

    def render_and_record(self, action_32):
        """Helper to render cams, log to Rerun, and add frames to recorder."""
        rgb_top = self.world_cam_top.render()[0]
        rgb_left = self.world_cam_left.render()[0]
        rgb_right = self.world_cam_right.render()[0]
        rgb_center = self.world_cam_center.render()[0]
        rgb_wrist = self.world_cam_wrist.render()[0]

        # Log visualization to Rerun
        rr.log("world_top", rr.Image(rgb_top[..., :3]))
        rr.log("world_left", rr.Image(rgb_left[..., :3]))
        rr.log("world_right", rr.Image(rgb_right[..., :3]))
        rr.log("world_center", rr.Image(rgb_center[..., :3]))
        rr.log("world_wrist", rr.Image(rgb_wrist[..., :3]))

        # Save to dataset if recording
        if self.is_recording:
            imgs = {
                "world_top": rgb_top,
                "world_left": rgb_left,
                "world_right": rgb_right,
                "world_center": rgb_center,
                "world_wrist": rgb_wrist,
            }
            self.recorder.add_frame(imgs, self.get_state_32(), action_32)

    def dispatch_action(
        self, num_steps, num_render_steps, action_32, target_q, start_q=None
    ):
        """
        Executes a physics burst.
        If start_q is provided, it performs a smooth interpolation.
        Otherwise, it holds target_q for the duration.
        """
        for idx in range(num_steps):
            if start_q is not None:
                # Smooth interpolation (glide)
                alpha = idx / float(num_steps)
                current_q = start_q + alpha * (target_q - start_q)
            else:
                # Hold fixed target (manual step)
                current_q = target_q

            self.robot.control_dofs_position(current_q)
            self.scene.step()

            # Render and record based on num_render_steps
            if idx % num_render_steps == 0:
                self.render_and_record(action_32)

    def run(self, port=5556):
        """Main Loop: Blocking ZMQ (REP) -> Physics Burst"""
        if self.test_mode:
            print("[TEST] Skipping ZMQ and Rerun initialization in run().")
            return

        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{port}")

        rr.init("gr1_teleop", spawn=False)
        # Handle connection failures gracefully if proxy isn't up
        try:
            rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        except Exception as e:
            print(f"⚠️ Rerun proxy not found: {e}")

        print("\n🚀 BLOCKING SIMULATION RUNNING (Single-Threaded)")
        print("  - Mode: Sequential (Draining -> Step 500 -> Wait)")
        print("  - Gains: KP=3500, KV=150 (Stable Precision)")
        print("  - Force: 500 Nm (MAX POWER)\n")

        while self.is_running:
            # Block and wait for a UI message
            msg = socket.recv()

            data = msgpack.unpackb(msg, raw=False)
            cmd = data.get("command")
            if cmd == "reset":
                self.reset_env()
                self.scene.step()
                socket.send(
                    msgpack.packb(
                        {"status": "reset_ok", "joints": self.get_state_32().tolist()}
                    )
                )
                continue
            elif cmd == "start_recording":
                self.current_task = data.get("task", "Pick up the red cube")
                self.recorder.start_episode(self.current_task)
                self.is_recording = True
                socket.send(msgpack.packb({"status": "recording_started"}))
                continue
            elif cmd == "auto_reach":
                offset_cm = data.get("offset_cm", 20.0)  # Handle configurable height
                target_z_offset = offset_cm / 100.0  # to meters

                # Auto reach sequence
                print(
                    f"[AUTO-REACH] Moving hand {offset_cm}cm above cube...", flush=True
                )
                cube_pos = self.cube.get_pos().cpu().numpy()
                target_pos = cube_pos + np.array([0.0, 0.0, target_z_offset])
                target_quat = [0.707, 0.0, 0.707, 0.0]  # Top-down

                start_q = self.robot.get_qpos().clone()
                target_q = self.solve_ik(target_pos, target_quat)

                # Precompute final action vector for recording
                target_q_32 = torch.stack(
                    [target_q[m["dof_idx"]] for m in self.joint_dof_map]
                )
                final_action_32 = self._normalize_state(target_q_32).cpu().numpy()

                # Unified Glide Burst (500 steps)
                self.dispatch_action(
                    500, 50, final_action_32, target_q, start_q=start_q
                )

                # Respond with the resulting 32-DOF joint vector for slider sync
                self.last_target_q = target_q.clone()
                resp_joints = self.get_state_32().tolist()

                socket.send(
                    msgpack.packb({"status": "auto_reach_ok", "joints": resp_joints})
                )
                continue
            elif cmd == "stop_recording":
                if self.is_recording:
                    self.recorder.stop_episode()
                    self.is_recording = False
                socket.send(msgpack.packb({"status": "recording_stopped"}))
                continue

            if "target" in data:
                self.active_joints_this_command.clear()
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)

                # Unified Target Burst (200 steps)
                print(f"Stepping physics (200Hz) for 1.0s sim-time...")
                self.dispatch_action(200, 20, action_32, self.last_target_q)

                # Final check of positions for all joints that received input
                print("\n[OUTPUT] Command Finished. Active Joints Status:")
                curr_q = self.robot.get_dofs_position().cpu().numpy()
                for idx in sorted(list(self.active_joints_this_command)):
                    mapping = self.joint_dof_map[idx]
                    joint_name = mapping["name"]
                    dof_idx = mapping["dof_idx"]
                    target = self.last_target_q[dof_idx]
                    actual = curr_q[dof_idx]
                    diff = actual - target
                    print(
                        f"  {joint_name:<30} | Tar: {target:6.3f} | Act: {actual:6.3f} | Err: {diff:6.3f}"
                    )
                print("------------------------------------------\n")

                # Send confirmation back to REQ socket
                socket.send(msgpack.packb({"status": "step_ok"}))


if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
