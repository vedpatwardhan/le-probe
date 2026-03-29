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
        self.world_cam_right = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(0.3, -1.5, 1.2), lookat=(0.3, 0, 0.8)
        )
        self.world_cam_left = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(0.3, 1.5, 1.2), lookat=(0.3, 0, 0.8)
        )
        self.world_cam_center = self.scene.add_camera(
            res=(224, 224), fov=60, pos=(1.5, 0.0, 1.2), lookat=(0, 0, 0.8)
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

        # Build joint DOF mapping (Must match Simulation.py exactly)
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
                    if (
                        other_joint.name
                        and finger_prefix in other_joint.name
                        and other_joint.name != joint_name
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

        # 1. Stiff Gains (Matched to teleop precision)
        self.robot.set_dofs_kp(np.full(self.robot.n_dofs, 3500.0))
        self.robot.set_dofs_kv(np.full(self.robot.n_dofs, 150.0))

        # 2. Neutral Start (Set to zeros before first step)
        self.robot.set_dofs_position(np.zeros(self.robot.n_dofs, dtype=np.float32))
        self.scene.step()

    def _normalize_state(self, raw_state):
        """Normalizes raw joint positions to [-1, 1] based on joint limits."""
        # Calculate range safely to avoid division by zero
        joint_range = JOINT_LIMITS_MAX - JOINT_LIMITS_MIN

        # Where range is very small (< 1e-4), treat as fixed/zero to avoid nuclear expansion
        safe_range = np.where(joint_range > 1e-4, joint_range, 1e-4)

        normalized_state = (raw_state - JOINT_LIMITS_MIN) / safe_range * 2.0 - 1.0

        # Hard clip to prevent any uninitialized or buggy values from poisoning the dataset
        return np.clip(normalized_state, -2.0, 2.0)

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

        return self._normalize_state(raw_state)

    def apply_action_32(self, action_32):
        """Action is normalized [-1, 1], unnormalize to Radians then apply.
        Strict protocol implementation for GR00T-N1.5.
        """
        logging.info(
            f"[ZMQ_ACTION_STATS] min: {float(np.min(action_32)):.3f}, max: {float(np.max(action_32)):.3f}, mean: {float(np.mean(action_32)):.3f}"
        )
        target_q = self.robot.get_dofs_position().cpu().numpy()

        # ONLY update the joints that were actually fine-tuned (Teleop Set)
        # 16-18 (R-Shoulder), 23-28 (R-Hand), 29-30 (Waist) - [Wrist Roll 21 Removed]
        TELEOP_INDICES = [16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30]

        for idx, mapping in enumerate(self.joint_dof_map):
            if idx not in TELEOP_INDICES:
                # Keep other joints at zero (Homed) to avoid noise
                target_q[mapping["dof_idx"]] = 0.0
                continue

            val = action_32[idx]
            val = np.clip(val, -1.0, 1.0)
            limit_min, limit_max = mapping["limits"]
            target_rad = (val + 1.0) / 2.0 * (limit_max - limit_min) + limit_min
            target_q[mapping["dof_idx"]] = target_rad

            # Apply finger coupling (proximal -> distal)
            for c_idx in mapping["coupled"]:
                target_q[c_idx] = target_rad

        # Apply the final computed 32-dim target position
        self.robot.control_dofs_position(target_q)

    def run(self, instruction="Pick up the red cube"):
        client = GR1Client()
        rr.init("gr1_sim", spawn=False)
        rr.connect_grpc(
            "rerun+http://neonh-103-96-40-120.a.free.pinggy.link:34069/proxy"
        )
        logging.info(f"Starting Multi-Step Inference Task: {instruction}")

        # Outer Loop: Number of inference requests to make (75 * 1.6s = 120 seconds)
        for cycle in range(75):
            logging.info(f"--- Inference Cycle {cycle+1}/75 ---")

            # Perception: Capture observation at the start of the horizon
            rgb_top, _, _, _ = self.world_cam_top.render()
            rgb_left, _, _, _ = self.world_cam_left.render()
            rgb_right, _, _, _ = self.world_cam_right.render()
            rgb_center, _, _, _ = self.world_cam_center.render()
            rgb_wrist, _, _, _ = self.world_cam_wrist.render()

            state_32 = self.get_state_32()
            obs = {
                "world_top": rgb_top[..., :3],
                "world_left": rgb_left[..., :3],
                "world_right": rgb_right[..., :3],
                "world_center": rgb_center[..., :3],
                "world_wrist": rgb_wrist[..., :3],
                "state": state_32,
                "instruction": instruction,
            }

            # Inference: Get 16-action chunk
            logging.info("Requesting Action Chunk...")
            actions = client.get_action_chunk(obs)  # returns list of 16 actions

            # Execution: Inner Loop (Each action is 0.1s / 20 physics steps)
            for action_idx, action in enumerate(actions):
                self.apply_action_32(np.array(action))

                # Step physics for 0.1s (50 steps @ 500Hz)
                for _ in range(50):
                    self.scene.step()

                # Render once per action step for 50Hz visual update
                rgb_top_step, _, _, _ = self.world_cam_top.render()
                rgb_left_step, _, _, _ = self.world_cam_left.render()
                rgb_right_step, _, _, _ = self.world_cam_right.render()
                rgb_center_step, _, _, _ = self.world_cam_center.render()
                rgb_wrist_step, _, _, _ = self.world_cam_wrist.render()

                rr.log("world_top", rr.Image(rgb_top_step[..., :3]))
                rr.log("world_left", rr.Image(rgb_left_step[..., :3]))
                rr.log("world_right", rr.Image(rgb_right_step[..., :3]))
                rr.log("world_center", rr.Image(rgb_center_step[..., :3]))
                rr.log("world_wrist", rr.Image(rgb_wrist_step[..., :3]))

                # Log progress
                total_actions = cycle * len(actions) + action_idx
                if total_actions % 10 == 0:
                    logging.info(f"Executed total actions: {total_actions}")

        logging.info("Task Sequence Complete.")


if __name__ == "__main__":
    sim = GR1Simulation()
    sim.run()
