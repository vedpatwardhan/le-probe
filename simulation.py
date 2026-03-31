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
            sim_options=gs.options.SimOptions(dt=0.0025, substeps=1),
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

        # Absolute Aggressive Control Profile (Optimized for maximum acceleration)
        kp_array = np.full(self.robot.n_dofs, 80000.0)
        kv_array = np.full(self.robot.n_dofs, 1200.0)
        self.robot.set_dofs_kp(kp_array)
        self.robot.set_dofs_kv(kv_array)

        # Industrial Force (MAX POWER: 5,000 Nm)
        force_max = np.full(self.robot.n_dofs, 5000.0)
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

        # Pre-computed protocol mapping tensors for Parallel Vectorization
        self.v_proto_indices = torch.arange(32, device=DEVICE)
        self.v_dof_indices = torch.zeros(32, device=DEVICE, dtype=torch.long)
        self.v_joint_names = COMPACT_WIRE_JOINTS

        # Collect finger coupling graph (Proximal Dofs -> Target Dofs)
        coupling_sources = []
        coupling_targets = []

        for idx, joint_name in enumerate(COMPACT_WIRE_JOINTS):
            joint = self.robot.get_joint(joint_name)
            dof_idx = joint.dofs_idx[0]
            self.v_dof_indices[idx] = dof_idx

            # Analyze Finger Coupling graph for this protocol joint
            if "proximal" in joint_name.lower():
                finger_prefix = joint_name.split("_proximal")[0]
                for other_joint in self.robot.joints:
                    if other_joint.name and finger_prefix in other_joint.name:
                        if (
                            other_joint.name != joint_name
                            and "proximal" not in other_joint.name.lower()
                        ):
                            # This intermediate/distal joint mimics its proximal protocol parent
                            coupling_sources.append(dof_idx)  # Source DOFs
                            coupling_targets.append(other_joint.dofs_idx[0])

        self.v_coupling_sources = torch.tensor(
            coupling_sources, device=DEVICE, dtype=torch.long
        )
        self.v_coupling_targets = torch.tensor(
            coupling_targets, device=DEVICE, dtype=torch.long
        )

        # Vectorized Authorization Mask (1.0 for authorized, 0.0 for ignored)
        self.v_allowed_mask = torch.zeros(32, device=DEVICE, dtype=torch.float32)

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
                self.v_allowed_mask[i] = 1.0

        # Also maintain raw DOF indices for IK solver
        self.ik_dof_indices = []
        self.v_ik_dof_indices_local = []

        # Calculate base offset for local indexing
        base_dof_idx = self.robot.q_start
        for j in self.robot.joints:
            if j.name in self.allowed_names_set:
                self.ik_dof_indices.append(j.dofs_idx[0])
                self.v_ik_dof_indices_local.append(j.dofs_idx[0] - base_dof_idx)

        print(
            f"✅ Unified Mask initialized with {len(self.allowed_32_indices)} authorized joints."
        )

    @property
    def joint_dof_map(self):
        """Backward compatibility for diagnostic tests; do not use in the control loop."""
        legacy_map = []
        for i, name in enumerate(self.v_joint_names):
            dof_idx = int(self.v_dof_indices[i])
            # Reconstruct coupling list from vectorized maps
            coupled = []
            if len(self.v_coupling_targets) > 0:
                mask = self.v_coupling_sources == dof_idx
                coupled = self.v_coupling_targets[mask].tolist()

            legacy_map.append(
                {
                    "dof_idx": dof_idx,
                    "name": name,
                    "limits": (float(self.wire_min[i]), float(self.wire_max[i])),
                    "coupled": coupled,
                }
            )
        return legacy_map
        for i in sorted(self.allowed_32_indices):
            mapping = self.joint_dof_map[i]
            print(
                f"  - Authorized Proxy [{i:02}] -> Robot DOF {mapping['dof_idx']} ({mapping['name']})"
            )

    def solve_ik(self, pos, quat):
        """
        Authoritative IK Solver: Constrains the Jacobian to native authorized joints.
        """
        # Get proposed full-body IK from Genesis, restricted to authorized group
        q_proposed = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=pos,
            quat=quat,
            dofs_idx_local=self.v_ik_dof_indices_local,
        )
        q_final = self.home_q.clone()

        # Batch Move IK shoulder/arm group (Overwrite home values with solver proposals)
        q_proposed_clamped = torch.clamp(
            q_proposed[self.ik_dof_indices],
            self.robot.get_dofs_limit()[0][self.ik_dof_indices],
            self.robot.get_dofs_limit()[1][self.ik_dof_indices],
        )
        q_final[self.ik_dof_indices] = q_proposed_clamped

        # Also preserve ALL currently authorized 32-DOF joints (Sliders)
        # This prevents snapping other sliders to home if they weren't in the IK group
        current_q = self.robot.get_qpos().clone()
        mask_indices = torch.tensor(
            list(self.allowed_32_indices), device=DEVICE, dtype=torch.long
        )
        dof_indices = self.v_dof_indices[mask_indices]
        q_final[dof_indices] = current_q[dof_indices]

        # Finally, overwrite specifically with the NEW IK results where intended
        q_final[self.ik_dof_indices] = q_proposed_clamped

        return q_final

    def get_state_32(self):
        """Returns the current 32-DOF joint positions as a NumPy array (Vectorized)."""
        raw_full_state = self.robot.get_dofs_position()
        # Extract and normalize all 32 joints in O(1)
        return self._normalize_state(raw_full_state[self.v_dof_indices]).cpu().numpy()

    def _normalize_state(self, raw_state):
        """
        Normalize 32-DOF protocol vector [-1.0, 1.0].
        Unauthorized joints are mapped to NaN in a single vectorized pass.
        """
        if not isinstance(raw_state, torch.Tensor):
            raw_state = torch.tensor(raw_state, device=DEVICE, dtype=torch.float32)

        # Parallel normalization across all 32 joints
        range_val = self.wire_max - self.wire_min
        range_val = torch.where(range_val > 1e-4, range_val, 1e-4)  # Div protection

        normalized = 2.0 * (raw_state - self.wire_min) / range_val - 1.0

        # unauthorized joints -> NaN
        masked = normalized.clone()
        masked[self.v_allowed_mask == 0] = float("nan")

        return masked.clamp(-1.1, 1.1)

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
                # Find joint name for this robot DOF
                joint_name = "unknown"
                for proxy_idx in self.allowed_32_indices:
                    if int(self.v_dof_indices[proxy_idx]) == i:
                        joint_name = self.v_joint_names[proxy_idx]
                        break

                print(
                    f"  DOF {i:02} ({joint_name:<35}) changed: "
                    f"{home_q[i]:.4f} -> {curr_q[i]:.4f}"
                )
                changed_count += 1
        print(f"Total Modified DOFs: {changed_count}")

        # Smooth Glide to Randomized Pose instead of teleporting
        self.last_target_q = q.clone()
        self.dispatch_action(
            np.full(32, np.nan), q, start_q=self.robot.get_qpos().clone()
        )
        print(
            f"[RESET] Env randomized. Cube: ({rx:.2f}, {ry:.2f}) | "
            f"Waist Pitch: {q[waist_pitch_joint.dofs_idx[0]]:.2f}"
        )

    def process_target_32(self, action_32):
        """Processes a 32-DOF action vector into joint targets (Vectorized)."""
        if not isinstance(action_32, torch.Tensor):
            action_32 = torch.tensor(action_32, device=DEVICE, dtype=torch.float32)

        # Authorize & Mask non-NaN joints (O(1))
        mask = (~torch.isnan(action_32)) & (self.v_allowed_mask > 0.5)
        if not mask.any():
            return False

        # Clipping happens implicitly if needed, or explicitly for safety:
        safe_action = torch.clamp(action_32[mask], -1.0, 1.0)

        range_val = self.wire_max - self.wire_min
        target_rads = (safe_action + 1.0) / 2.0 * range_val[mask] + self.wire_min[mask]

        # Parallel application to primary robot DOFs
        self.last_target_q[self.v_dof_indices[mask]] = target_rads

        # Parallel Finger Coupling (mimic proximal -> distal in O(1))
        if len(self.v_coupling_targets) > 0:
            self.last_target_q[self.v_coupling_targets] = self.last_target_q[
                self.v_coupling_sources
            ]

        # Reduced I/O diagnostic: Summary only
        non_nan_count = (~torch.isnan(action_32)).sum().item()
        print(f"[INPUT] Vectorized update for {non_nan_count} authorized joints.")
        return True

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

    def print_joint_status(self):
        """Prints high-fidelity Target vs. Actual diagnostics for all active joints."""
        if not self.active_joints_this_command:
            print("\n[OUTPUT] No Active Joints.")
            return

        print("\n[OUTPUT] Command Finished. Active Joints Status:")
        curr_q = self.robot.get_dofs_position().cpu().numpy()
        for idx in sorted(list(self.active_joints_this_command)):
            mapping = self.joint_dof_map[idx]
            joint_name = mapping["name"]
            dof_idx = mapping["dof_idx"]
            target = float(self.last_target_q[dof_idx])
            actual = float(curr_q[dof_idx])
            diff = actual - target
            print(
                f"  {joint_name:<35} | Tar: {target:6.3f} | Act: {actual:6.3f} | Err: {diff:6.3f}"
            )
        print("-" * 50 + "\n")

    def dispatch_action(self, action_32, target_q, start_q=None):
        """
        High-Frequency High-Authority Burst: 400 steps (1.0s at 400Hz).
        Recording: 10 frames (every 40 steps).
        """
        # Diagnostic Registration (Selection-Based Parity)
        if action_32 is not None:
            mask_nan = ~np.isnan(action_32)
            for idx in np.where(mask_nan)[0]:
                self.active_joints_this_command.add(int(idx))

        # Fallback Registration (Change-Based for Resets/IK)
        if not self.active_joints_this_command:
            current_robot_q = self.robot.get_dofs_position()
            ref_q = start_q if start_q is not None else current_robot_q
            diff = torch.abs(target_q - ref_q)
            changed_dof_indices = torch.where(diff > 1e-4)[0].tolist()
            for proxy_idx in self.allowed_32_indices:
                if int(self.v_dof_indices[proxy_idx]) in changed_dof_indices:
                    self.active_joints_this_command.add(proxy_idx)

        # 1.0s Fixed-Window Physics Burst (400 steps @ 400Hz)
        num_steps = 400
        num_render_steps = 40
        ref_start_q = start_q if start_q is not None else self.robot.get_qpos().clone()

        for idx in range(num_steps):
            alpha = idx / float(num_steps)
            current_q = ref_start_q + alpha * (target_q - ref_start_q)
            self.robot.control_dofs_position(current_q)
            self.scene.step()

            # Diagnostic rendering/recording sync (10 frames)
            if idx % num_render_steps == 0:
                self.render_and_record(action_32)

        # Final status after every physics burst (Diagnostic Parity)
        self.print_joint_status()

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
        print("  - Mode: Sequential (400Hz / 400 Steps)")
        print("  - Gains: KP=80000, KV=1200 (Absolute Aggressive)")
        print("  - Force: 5000 Nm (ULTRA POWER)\n")

        while self.is_running:
            # Block and wait for a UI message
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            cmd = data.get("command")

            # Prep for diagnostic tracking
            self.active_joints_this_command.clear()

            if cmd == "reset":
                self.reset_env()
                self.scene.step()
                socket.send(
                    msgpack.packb(
                        {
                            "status": "reset_ok",
                            "joints": self.get_state_32().tolist(),
                            "upload_queue": self.recorder.pending_uploads,
                            "total_episodes": self.recorder.total_episodes,
                        }
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

                # Respond with the resulting 32-DOF joint vector for slider sync
                self.last_target_q = target_q.clone()
                self.dispatch_action(final_action_32, target_q, start_q=start_q)
                resp_joints = self.get_state_32().tolist()

                socket.send(
                    msgpack.packb(
                        {
                            "status": "auto_reach_ok",
                            "joints": resp_joints,
                            "upload_queue": self.recorder.pending_uploads,
                            "total_episodes": self.recorder.total_episodes,
                        }
                    )
                )
                continue
            elif cmd == "stop_recording":
                if self.is_recording:
                    self.recorder.stop_episode()
                    self.is_recording = False
                socket.send(
                    msgpack.packb(
                        {
                            "status": "recording_stopped",
                            "upload_queue": self.recorder.pending_uploads,
                            "total_episodes": self.recorder.total_episodes,
                        }
                    )
                )
                continue
            elif cmd == "poll_status":
                socket.send(
                    msgpack.packb(
                        {
                            "status": "status_ok",
                            "upload_queue": self.recorder.pending_uploads,
                            "total_episodes": self.recorder.total_episodes,
                        }
                    )
                )
                continue

            if "target" in data:
                action_32 = np.array(data["target"], dtype=np.float32)
                self.process_target_32(action_32)

                # Unified Target Burst (Uniform: 1.0s)
                print(f"Stepping physics (200Hz) for 1.0s sim-time...")
                self.dispatch_action(action_32, self.last_target_q)

                # Send confirmation back to REQ socket
                socket.send(
                    msgpack.packb(
                        {
                            "status": "step_ok",
                            "upload_queue": self.recorder.pending_uploads,
                            "total_episodes": self.recorder.total_episodes,
                        }
                    )
                )


if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
