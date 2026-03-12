import numpy as np
import genesis as gs
import rerun as rr
import zmq
import msgpack
import logging
import time
from scipy.spatial.transform import Rotation as R
import torch
import xml.etree.ElementTree as ET
from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    CAMERA_ATTACH_LINK,
)

URDF_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/repos/Wiki-GRx-Models/GRX/GR1/gr1t2/urdf/gr1t2_fourier_hand_6dof.urdf"

# -----------------------------------------------------------------------------
# CONFIG & LOGGING
# -----------------------------------------------------------------------------
# We mute logging because this script will run as a background subprocess
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GR1Simulation:
    def __init__(self, urdf_path=URDF_PATH):
        # Initialize Genesis
        gs.init(backend=gs.gpu)

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

        # Add cameras for top, left, right and center views
        self.world_cam_top = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(0.3, 0, 1.8), lookat=(0.3, 0, 0.8)
        )
        self.world_cam_left = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.0, -1.0, 1.2), lookat=(0, 0, 0.8)
        )
        self.world_cam_right = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.0, 1.0, 1.2), lookat=(0, 0, 0.8)
        )
        self.world_cam_center = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.0, 0.0, 1.2), lookat=(0, 0, 0.8)
        )

        # Build the scene
        self.scene.build()

        # DEBUG: Print all joints found in the robot to verify naming
        print("\n--- Genesis Robot Joint Discovery ---")
        for i, joint in enumerate(self.robot.joints):
            print(
                f"[{i}] Name: {joint.name} | Type: {joint.type} | DOFs: {len(joint.dofs_idx_local)}"
            )
        print("--------------------------------------\n")

        # Gains & Control
        # Higher gains for major joints, lower for fingers to prevent simulation instability
        # Arm/Body Gains
        kp_array = np.full(self.robot.n_dofs, 450.0)
        kv_array = np.full(self.robot.n_dofs, 45.0)

        # Fingers: Extreme stiffness to overcome URDF friction/collisions
        for joint in self.robot.joints:
            name = joint.name
            if name and any(
                f in name.lower() for f in ["thumb", "index", "middle", "ring", "pinky"]
            ):
                for i in joint.dofs_idx:  # FIXED: Use global DOF index
                    kp_array[i] = 5000.0
                    kv_array[i] = 1.0
                    print(
                        f"[DEBUG] Finger Gain Assigned: {name} (Global DOF {i}) -> KP=5000"
                    )

        self.robot.set_dofs_kp(kp_array)
        self.robot.set_dofs_kv(kv_array)

        # Force Range (Effort Limits Override)
        # The URDF has effort=0 for fingers, which blocks movement.
        # We override this in software to give them 'muscle'.
        force_max = np.full(self.robot.n_dofs, 100.0)  # High default for arms
        force_min = -force_max

        # Specific finger force range
        for joint in self.robot.joints:
            name = joint.name
            if name and any(
                f in name.lower() for f in ["thumb", "index", "middle", "ring", "pinky"]
            ):
                for i in joint.dofs_idx:
                    force_max[i] = 10.0
                    force_min[i] = -10.0

        self.robot.set_dofs_force_range(force_min, force_max)

        # Internal state
        self.target_buffer = np.full(32, np.nan, dtype=np.float32)
        # Persistent target state (Initializes all joints to 0.0)
        self.last_target_q = np.zeros(self.robot.n_dofs, dtype=np.float32)

        self.is_running = True
        self.active_joints_this_command = set()

        # Pre-index mappings for speed
        self._joint_dof_map = []
        for idx, joint_name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                joint = self.robot.get_joint(joint_name)
                dof_idx = joint.dofs_idx[
                    0
                ]  # CRITICAL FIX: Use global DOF index, not local
                limit_min, limit_max = JOINT_LIMITS_MIN[idx], JOINT_LIMITS_MAX[idx]

                # Finger coupling pre-indexing
                coupled_dofs = []
                if "proximal" in joint_name.lower():
                    finger_prefix = joint_name.split("_proximal")[0]
                    for other_joint in self.robot.joints:
                        if other_joint.name and finger_prefix in other_joint.name:
                            if (
                                other_joint.name != joint_name
                                and "proximal" not in other_joint.name.lower()
                            ):
                                coupled_dofs.append(
                                    other_joint.dofs_idx[0]
                                )  # Global index here too

                self._joint_dof_map.append(
                    {
                        "dof_idx": dof_idx,
                        "limits": (limit_min, limit_max),
                        "name": joint_name,
                        "coupled": coupled_dofs,
                    }
                )
            except:
                self._joint_dof_map.append(None)

    def process_target_32(self, action_32):
        """Maps 32-DOF UI actions to the full robot target state (once per message)."""
        any_update = False
        print(f"\n[INPUT] Received ZMQ message (32 DOFs):")
        for idx, mapping in enumerate(self._joint_dof_map):
            if mapping is None:
                continue

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
                self.last_target_q[dof_idx] = target_rad
                # Apply coupling
                for c_idx in mapping["coupled"]:
                    self.last_target_q[c_idx] = target_rad
                any_update = True

        if any_update:
            # We don't call control here; we call it inside the run loop for persistence
            pass
        print(f"[INPUT] Total Target State Updated: {any_update}\n")
        return any_update

    def run(self, port=5556):
        """Runs the simulation as a blocking server (waits for UI input)."""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.bind(f"tcp://127.0.0.1:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

        print(f"Simulation Process Ready. Listening on port {port}")
        print("Waiting for TUI command to step...")

        while self.is_running:
            # 1. Block until a new command is received
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            if "target" in data:
                self.target_buffer = np.array(data["target"], dtype=np.float32)

            # 2. Process message ONCE to update the target state
            self.active_joints_this_command.clear()
            self.process_target_32(self.target_buffer)

            # 3. Physics Step (1000 steps @ 0.002s = 2 seconds of sim time)
            # We call control_dofs_position EVERY step to ensure the motor stays active.
            print(f"[EXEC] Stepping physics (500Hz) for 2.0s sim-time...")
            for i in range(1000):
                self.robot.control_dofs_position(self.last_target_q)
                self.scene.step()

                # Render & Log a few frames for visual feedback (throttled)
                if i % 250 == 0:
                    # Visual Render
                    rgb_center, _, _, _ = self.world_cam_center.render()
                    rr.log("world_center", rr.Image(rgb_center[..., :3]))

                    # Progress Logging
                    print(f"  [Step {i:04}] Progress Check:")
                    curr_q_loop = self.robot.get_dofs_position().cpu().numpy()
                    for idx in sorted(list(self.active_joints_this_command)):
                        mapping = self._joint_dof_map[idx]
                        if mapping:
                            dof_idx = mapping["dof_idx"]
                            act = curr_q_loop[dof_idx]
                            tar = self.last_target_q[dof_idx]
                            print(
                                f"    {mapping['name']:<30} | Act: {act:6.3f} (Tar: {tar:6.3f})"
                            )

            # Final Render of ALL cameras at the end of the action
            rgb_top, _, _, _ = self.world_cam_top.render()
            rgb_left, _, _, _ = self.world_cam_left.render()
            rgb_right, _, _, _ = self.world_cam_right.render()
            rgb_center, _, _, _ = self.world_cam_center.render()

            rr.log("world_top", rr.Image(rgb_top[..., :3]))
            rr.log("world_left", rr.Image(rgb_left[..., :3]))
            rr.log("world_right", rr.Image(rgb_right[..., :3]))
            rr.log("world_center", rr.Image(rgb_center[..., :3]))

            # Final check of positions for all joints that received input
            print("\n[DEBUG] Command Finished. Active Joints Status:")
            curr_q = self.robot.get_dofs_position().cpu().numpy()

            if not self.active_joints_this_command:
                print("  No joints were updated in this command.")
            else:
                # Get current physics state
                curr_kp = self.robot.get_dofs_kp().cpu().numpy()
                curr_kv = self.robot.get_dofs_kv().cpu().numpy()

                for idx in sorted(list(self.active_joints_this_command)):
                    mapping = self._joint_dof_map[idx]
                    if mapping:
                        joint_name = mapping["name"]
                        dof_idx = mapping["dof_idx"]
                        target = self.last_target_q[dof_idx]
                        actual = curr_q[dof_idx]
                        diff = actual - target
                        kp = curr_kp[dof_idx]
                        kv = curr_kv[dof_idx]
                        l_min, l_max = mapping["limits"]
                        print(
                            f"  {joint_name:<30} | Tar: {target:6.3f} | Act: {actual:6.3f} | Err: {diff:6.3f} | KP: {kp:7.1f} | KV: {kv:5.1f}"
                        )
            print("------------------------------------------\n")


if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
