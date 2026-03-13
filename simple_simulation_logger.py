import numpy as np
import genesis as gs
import zmq
import msgpack
import msgpack_numpy as m
import logging
import torch
import time
import rerun as rr
from gr1_config import (
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    URDF_PATH,
)

m.patch()  # Enable msgpack to serialize numpy arrays directly

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GR1SimulationLogger:
    def __init__(self, urdf_path=URDF_PATH):
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
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

        # Cameras (224x224 typical for foundation models)
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

        self.scene.build()

        self.robot.set_dofs_kp(np.full(self.robot.n_dofs, 450.0))
        self.robot.set_dofs_kv(np.full(self.robot.n_dofs, 45.0))

        self.target_buffer = np.zeros(32, dtype=np.float32)
        self.is_running = True

    def get_state_32(self):
        q = self.robot.get_dofs_position().cpu().numpy()
        raw_state = []
        for name in COMPACT_WIRE_JOINTS:
            joint = self.robot.get_joint(name)
            raw_state.append(q[joint.dofs_idx_local[0]])
        raw_state = np.array(raw_state, dtype=np.float32)

        normalized_state = (raw_state - JOINT_LIMITS_MIN) / (
            JOINT_LIMITS_MAX - JOINT_LIMITS_MIN + 1e-8
        ) * 2.0 - 1.0
        return normalized_state

    def apply_action_32(self, action_32):
        target_q = np.zeros(self.robot.n_dofs, dtype=np.float32)
        # target_q = self.robot.get_dofs_position().cpu().numpy()
        for idx, joint_name in enumerate(COMPACT_WIRE_JOINTS):
            val = action_32[idx]
            if np.isnan(val):
                continue
            limit_min, limit_max = JOINT_LIMITS_MIN[idx], JOINT_LIMITS_MAX[idx]
            val = np.clip(val, -1.0, 1.0)
            target_rad = (val + 1.0) / 2.0 * (limit_max - limit_min) + limit_min
            joint = self.robot.get_joint(joint_name)
            target_q[joint.dofs_idx_local[0]] = target_rad
        self.robot.control_dofs_position(target_q)

    def run(self, sub_port=5556, pub_port=5557):
        context = zmq.Context()

        # SUB socket: Action Commands from TUI
        sub_socket = context.socket(zmq.SUB)
        sub_socket.bind(f"tcp://127.0.0.1:{sub_port}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # PUB socket: Observation streams strictly to TUI
        pub_socket = context.socket(zmq.PUB)
        pub_socket.bind(f"tcp://127.0.0.1:{pub_port}")

        # Rerun setup for visualization
        rr.init("gr1_teleop", spawn=False)
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")

        print(f"Simulation Logger Ready.")
        print(
            f"Listening for Actions on {sub_port} | Publishing Observations on {pub_port}"
        )
        print("Waiting for TUI command to step...")

        while self.is_running:
            # Block until a new command is received
            msg = sub_socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            if "target" in data:
                self.target_buffer = np.array(data["target"], dtype=np.float32)

            # Once received, run 100 steps of physics
            for i in range(100):
                self.scene.step()
                self.apply_action_32(self.target_buffer)

                # Render & Publish periodically (every 10 steps)
                if i % 10 == 0:
                    # Render Cameras
                    rgb_top, _, _, _ = self.world_cam_top.render()
                    rgb_left, _, _, _ = self.world_cam_left.render()
                    rgb_right, _, _, _ = self.world_cam_right.render()
                    rgb_center, _, _, _ = self.world_cam_center.render()

                    # Log to Rerun
                    rr.log("world_top", rr.Image(rgb_top[..., :3].astype(np.uint8)))
                    rr.log("world_left", rr.Image(rgb_left[..., :3].astype(np.uint8)))
                    rr.log("world_right", rr.Image(rgb_right[..., :3].astype(np.uint8)))
                    rr.log(
                        "world_center", rr.Image(rgb_center[..., :3].astype(np.uint8))
                    )

                    # Get State
                    state_32 = self.get_state_32()

                    # Publish observations back
                    obs_payload = {
                        "world_top": rgb_top[..., :3].astype(np.uint8),
                        "world_left": rgb_left[..., :3].astype(np.uint8),
                        "world_right": rgb_right[..., :3].astype(np.uint8),
                        "world_center": rgb_center[..., :3].astype(np.uint8),
                        "state": state_32,
                    }
                    pub_socket.send(msgpack.packb(obs_payload, use_bin_type=True))


if __name__ == "__main__":
    sim = GR1SimulationLogger()
    sim.run()
