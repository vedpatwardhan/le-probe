import numpy as np
import genesis as gs
import rerun as rr
import zmq
import msgpack
import logging
import os
import torch
from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_STATS,
    # URDF_PATH,
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

        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720), camera_pos=(2.0, 1.5, 1.5), camera_lookat=(0, 0, 0.8)
            ),
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

        # Entities
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=True, pos=(0, 0, 0.95))
        )

        # World Cameras (Diagnostic)
        self.world_cam_left = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.2, -1.2, 1.2), lookat=(0, 0, 0.8)
        )
        self.world_cam_right = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.2, 1.2, 1.2), lookat=(0, 0, 0.8)
        )

        # Build Scene
        self.scene.build()

        # Gains & Control
        self.robot.set_dofs_kp(np.full(self.robot.n_dofs, 450.0))
        self.robot.set_dofs_kv(np.full(self.robot.n_dofs, 45.0))

        # Internal state
        self.target_buffer = np.full(32, np.nan, dtype=np.float32)
        self.is_running = True

    def apply_action_32(self, action_32):
        """Action is normalized [-1, 1], unnormalize to Radians then apply.
        If a value is np.nan, it stays at its current position.
        """
        # Start with current degrees of freedom as the base
        target_q = np.zeros(self.robot.n_dofs, dtype=np.float32)
        # target_q = self.robot.get_dofs_position().cpu().numpy()

        for idx, joint_name in enumerate(COMPACT_WIRE_JOINTS):
            val = action_32[idx]
            if np.isnan(val):
                continue

            # Unnormalize the target for this specific joint
            print(idx)
            limit_min, limit_max, _ = JOINT_STATS[idx]
            val = np.clip(val, -1.0, 1.0)
            target_rad = (val + 1.0) / 2.0 * (limit_max - limit_min) + limit_min

            joint = self.robot.get_joint(joint_name)
            target_q[joint.dofs_idx_local[0]] = target_rad

        print(target_q)
        self.robot.control_dofs_position(target_q)

    def run(self, port=5556):
        """Runs the simulation as a synchronous ZMQ server."""
        # 1. Setup ZMQ SUB socket to receive targets
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.bind(f"tcp://127.0.0.1:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all

        # 2. Setup Rerun
        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

        print(f"Simulation Process Ready. Listening on port {port}")
        print("Waiting for TUI command to step...")

        while self.is_running:
            # Block until a new command is received
            msg = socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            if "target" in data:
                self.target_buffer = np.array(data["target"], dtype=np.float32)

            # Once received, run 100 steps of physics
            for i in range(100):
                self.scene.step()
                self.apply_action_32(self.target_buffer)

                # Render & Log periodically (every 10 steps) to show motion
                if i % 10 == 0:
                    rgb_left, _, _, _ = self.world_cam_left.render()
                    rgb_right, _, _, _ = self.world_cam_right.render()
                    rr.log("world_left", rr.Image(rgb_left[..., :3]))
                    rr.log("world_right", rr.Image(rgb_right[..., :3]))


if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
