import numpy as np
import genesis as gs
import rerun as rr
import zmq
import msgpack
import logging
import torch
import xml.etree.ElementTree as ET
from gr1_config import COMPACT_WIRE_JOINTS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX

URDF_PATH = "./repos/Wiki-GRx-Models/GRX/GR1/gr1t2/urdf/gr1t2_fourier_hand_6dof.urdf"

# -----------------------------------------------------------------------------
# CONFIG & LOGGING
# -----------------------------------------------------------------------------
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

        # Pre-index mappings for speed
        self._joint_dof_map = []
        for idx, joint_name in enumerate(COMPACT_WIRE_JOINTS):
            try:
                joint = self.robot.get_joint(joint_name)
                dof_idx = joint.dofs_idx[0]
                limit_min, limit_max = JOINT_LIMITS_MIN[idx], JOINT_LIMITS_MAX[idx]

                # Finger coupling
                coupled_dofs = []
                if "proximal" in joint_name.lower():
                    finger_prefix = joint_name.split("_proximal")[0]
                    for other_joint in self.robot.joints:
                        if other_joint.name and finger_prefix in other_joint.name:
                            if other_joint.name != joint_name and "proximal" not in other_joint.name.lower():
                                coupled_dofs.append(other_joint.dofs_idx[0])

                self._joint_dof_map.append({
                    "dof_idx": dof_idx,
                    "limits": (limit_min, limit_max),
                    "name": joint_name,
                    "coupled": coupled_dofs,
                })
            except:
                self._joint_dof_map.append(None)

    def process_target_32(self, action_32):
        """Maps 32-DOF actions to targets and logs progress."""
        any_update = False
        print(f"\n[INPUT] Received ZMQ message (32 DOFs):")
        for idx, mapping in enumerate(self._joint_dof_map):
            if mapping is None: continue
            val = action_32[idx]
            if np.isnan(val): continue

            self.active_joints_this_command.add(idx)
            val = np.clip(val, -1.0, 1.0)
            limit_min, limit_max = mapping["limits"]
            target_rad = (val + 1.0) / 2.0 * (limit_max - limit_min) + limit_min

            dof_idx = mapping["dof_idx"]
            print(f"  [{idx:02}] {mapping['name']:<30} | In: {val:6.3f} -> Tar: {target_rad:6.3f} rad")

            if abs(self.last_target_q[dof_idx] - target_rad) > 1e-4:
                self.last_target_q[dof_idx] = target_rad
                for c_idx in mapping["coupled"]:
                    self.last_target_q[c_idx] = target_rad
                any_update = True
        return any_update

    def run(self, port=5556):
        """Main Loop: Blocking ZMQ -> Physics Burst"""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.bind(f"tcp://127.0.0.1:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

        print("\n🚀 BLOCKING SIMULATION RUNNING (Single-Threaded)")
        print("  - Mode: Sequential (Draining -> Step 500 -> Wait)")
        print("  - Gains: KP=3500, KV=150 (Stable Precision)")
        print("  - Force: 500 Nm (MAX POWER)\n")

        while self.is_running:
            # 1. Block and wait for at least one UI message
            msg = socket.recv() 
            
            # Drain buffer to get ONLY the latest message (prevents repeating old actions)
            while True:
                try:
                    msg = socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
            
            data = msgpack.unpackb(msg, raw=False)
            if "target" in data:
                self.active_joints_this_command.clear()
                target = np.array(data["target"], dtype=np.float32)
                self.process_target_32(target)

                # 2. Physics Burst (100 steps @ 0.01s = 1.0 second of sim-time)
                print(f"[EXEC] Stepping physics (100Hz) for 1.0s sim-time...")
                for i in range(100):
                    self.robot.control_dofs_position(self.last_target_q)
                    self.scene.step()

                    # Render every 20 steps (~25 FPS feedback)
                    if i % 20 == 0:
                        rgb_top = self.world_cam_top.render()[0]
                        rgb_left = self.world_cam_left.render()[0]
                        rgb_right = self.world_cam_right.render()[0]
                        rgb_center = self.world_cam_center.render()[0]

                        rr.log("world_top", rr.Image(rgb_top[..., :3]))
                        rr.log("world_left", rr.Image(rgb_left[..., :3]))
                        rr.log("world_right", rr.Image(rgb_right[..., :3]))
                        rr.log("world_center", rr.Image(rgb_center[..., :3]))

                # 3. Final check of positions for all joints that received input
                print("\n[DEBUG] Command Finished. Active Joints Status:")
                curr_q = self.robot.get_dofs_position().cpu().numpy()
                for idx in sorted(list(self.active_joints_this_command)):
                    mapping = self._joint_dof_map[idx]
                    if mapping:
                        joint_name = mapping["name"]
                        dof_idx = mapping["dof_idx"]
                        target = self.last_target_q[dof_idx]
                        actual = curr_q[dof_idx]
                        diff = actual - target
                        print(f"  {joint_name:<30} | Tar: {target:6.3f} | Act: {actual:6.3f} | Err: {diff:6.3f}")
                print("------------------------------------------\n")

if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
