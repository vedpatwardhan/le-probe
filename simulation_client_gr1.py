import numpy as np
import genesis as gs
import rerun as rr
import zmq
import msgpack
import logging
from scipy.spatial.transform import Rotation as R
import torch
from gr1_config import (
    CAMERA_ATTACH_LINK,
    COMPACT_WIRE_JOINTS,
    JOINT_LIMITS_MIN,
    JOINT_LIMITS_MAX,
    URDF_PATH,
)

# -----------------------------------------------------------------------------
# CONFIG & LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# GR00T CLIENT
# -----------------------------------------------------------------------------
class GR1Client:
    """Minimal ZMQ Client for GR00T Inference."""

    def __init__(self, host="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)
        self.socket.connect(f"tcp://{host}:{port}")

    def get_action_chunk(self, obs):
        payload = {
            k: (
                {
                    "__is_np_bytes__": True,
                    "data": v.tobytes(),
                    "shape": v.shape,
                    "dtype": str(v.dtype),
                }
                if isinstance(v, np.ndarray)
                else v
            )
            for k, v in obs.items()
        }
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        resp = msgpack.unpackb(self.socket.recv(), raw=False)
        return resp.get("action", [])


# -----------------------------------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------------------------------
class GR1Simulation:
    def __init__(self, urdf_path=URDF_PATH):
        gs.init(backend=gs.gpu, precision="32", logging_level="info")
        self._init_genesis(urdf_path)

    def _init_genesis(self, urdf_path):
        # Create Scene
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

        # Add entities
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, pos=(-0.3, 0, 0.95), fixed=True)
        )
        self.table = self.scene.add_entity(
            gs.morphs.Box(pos=(0.45, 0, 0.4), size=(0.4, 0.8, 0.8), fixed=True)
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(pos=(0.45, -0.1, 0.82), size=(0.04, 0.04, 0.04), fixed=False),
            surface=gs.surfaces.Default(color=(1, 0, 0)),
        )

        # Add egocentric camera
        head_link = self.robot.get_link(CAMERA_ATTACH_LINK)
        self.cam = self.scene.add_camera(res=(224, 224), fov=90)

        # Tilt the egocentric camera towards the cube
        rot = (
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            @ R.from_euler("x", -15, degrees=True).as_matrix()
        )
        offset_T = np.eye(4)
        offset_T[:3, :3] = rot
        offset_T[:3, 3] = [0.12, 0.0, 0.08]
        self.cam.attach(head_link, offset_T=offset_T)

        # Add cameras for left, right and center views
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

        # Gains & Control
        self.robot.set_dofs_kp(np.full(self.robot.n_dofs, 450.0))
        self.robot.set_dofs_kv(np.full(self.robot.n_dofs, 45.0))

    def get_state_32(self):
        """Extracts 32-dim normalized state: [L_Arm(7), L_Hand(6), Head(3), R_Arm(7), R_Hand(6), Waist(3)]"""
        # Get current position (32-dim)
        q = self.robot.get_dofs_position().cpu().numpy()

        # Iterate over joints and get current state
        raw_state = []
        for name in COMPACT_WIRE_JOINTS:
            joint = self.robot.get_joint(name)
            raw_state.append(q[joint.dofs_idx_local[0]])
        raw_state = np.array(raw_state, dtype=np.float32)

        # Normalize to [-1, 1]
        normalized_state = (raw_state - JOINT_LIMITS_MIN) / (
            JOINT_LIMITS_MAX - JOINT_LIMITS_MIN + 1e-8
        ) * 2.0 - 1.0
        return normalized_state

    def apply_action_32(self, action_32):
        """Action is normalized [-1, 1], unnormalize to Radians then apply.
        Strict protocol implementation for GR00T-N1.5.
        """
        logging.info(
            f"[ZMQ_ACTION_STATS] min: {float(np.min(action_32)):.3f}, max: {float(np.max(action_32)):.3f}, mean: {float(np.mean(action_32)):.3f}"
        )

        # Unnormalize from [-1, 1] back to Radians
        target_radians = (action_32 + 1.0) / 2.0 * (
            JOINT_LIMITS_MAX - JOINT_LIMITS_MIN
        ) + JOINT_LIMITS_MIN
        target_radians = np.clip(target_radians, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)

        # Map the 32 targets into the full robot DOF vector
        target_q = np.zeros(self.robot.n_dofs, dtype=np.float32)
        for idx, name in enumerate(COMPACT_WIRE_JOINTS):
            joint = self.robot.get_joint(name)
            target_q[joint.dofs_idx_local[0]] = target_radians[idx]

        # Apply the final computed 32-dim target position
        self.robot.control_dofs_position(target_q)

    def run(self, instruction="Pick up the red cube from the table down there."):
        client = GR1Client()
        rr.init("gr1_sim", spawn=False)
        rr.connect_grpc(
            "rerun+http://wtnwo-103-96-40-120.a.free.pinggy.link:41009/proxy"
        )
        logging.info(f"Starting Multi-Step Inference Task: {instruction}")

        # Outer Loop: Number of inference requests to make
        for cycle in range(30):
            logging.info(f"--- Inference Cycle {cycle+1}/30 ---")

            # Perception: Capture observation at the start of the horizon
            self.cam.move_to_attach()
            rgb_ego, _, _, _ = self.cam.render()
            rgb_left, _, _, _ = self.world_cam_left.render()
            rgb_right, _, _, _ = self.world_cam_right.render()
            rgb_center, _, _, _ = self.world_cam_center.render()

            state_32 = self.get_state_32()
            obs = {
                "head": rgb_ego[..., :3],
                "world_left": rgb_left[..., :3],
                "world_right": rgb_right[..., :3],
                "world_center": rgb_center[..., :3],
                "state": state_32,
                "instruction": instruction,
            }

            # Inference: Get 16-action chunk
            logging.info("Requesting Action Chunk...")
            actions = client.get_action_chunk(obs)  # returns list of 16 actions

            # Execution: Inner Loop (16 actions, 10 physics steps each)
            for action_idx, action in enumerate(actions):
                self.apply_action_32(np.array(action))

                # Step physics for 0.1s (10 steps @ 100Hz)
                for _ in range(10):
                    self.scene.step()

                    # Render & Log ALL cameras to Rerun after every action step
                    self.cam.move_to_attach()
                    rgb_ego_step, _, _, _ = self.cam.render()
                    rgb_left_step, _, _, _ = self.world_cam_left.render()
                    rgb_right_step, _, _, _ = self.world_cam_right.render()
                    rgb_center_step, _, _, _ = self.world_cam_center.render()
                    rr.log("egocentric", rr.Image(rgb_ego_step[..., :3]))
                    rr.log("world_left", rr.Image(rgb_left_step[..., :3]))
                    rr.log("world_right", rr.Image(rgb_right_step[..., :3]))
                    rr.log("world_center", rr.Image(rgb_center_step[..., :3]))

                # Log progress
                if (cycle * 16 + action_idx) % 10 == 0:
                    logging.info(f"Executed total actions: {cycle * 16 + action_idx}")

        logging.info("Task Sequence Complete.")


if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
