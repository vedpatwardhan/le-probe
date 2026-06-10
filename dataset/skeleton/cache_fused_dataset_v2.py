"""
cache_fused_dataset_v2.py

High-speed direct dataset pre-cache compiler for Le-Probe v2.
Bypasses on-the-fly kinematics calculations by pre-computing and storing
the Chebyshev ellipsoid reachability parameters inside the serialized Torch files.
"""

import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import lz4.frame
import io
import mujoco

REPO_DIR = Path(__file__).resolve().parents[3]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Import required tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lewm.skeleton.dino_constants import validate_dino_waypoints, zeros_dino_waypoints
from gr1_config import SCENE_PATH
from gr1_protocol import StandardScaler


def compute_ellipsoid_analytically(model, data, dof_indices, ee_id, q_state):
    """Computes the ellipsoid parameters using SVD on-the-fly for caching."""
    scaler = StandardScaler()
    raw_state = scaler.unscale_action(
        q_state.numpy() if torch.is_tensor(q_state) else q_state
    )
    q_arm = raw_state[16:23]

    data.qpos[dof_indices] = q_arm
    mujoco.mj_forward(model, data)

    ee_pos = data.xpos[ee_id]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, ee_pos, ee_id)
    J_arm = jacp[:, dof_indices]

    U, S, Vt = np.linalg.svd(J_arm)
    c = np.zeros(3, dtype=np.float32)
    r = (0.2 * S).astype(np.float32)

    R = U
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S_val = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S_val
        qx = (R[2, 1] - R[1, 2]) / S_val
        qy = (R[0, 2] - R[2, 0]) / S_val
        qz = (R[1, 0] - R[0, 1]) / S_val
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S_val = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S_val
        qx = 0.25 * S_val
        qy = (R[0, 1] + R[1, 0]) / S_val
        qz = (R[0, 2] + R[2, 0]) / S_val
    elif R[1, 1] > R[2, 2]:
        S_val = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S_val
        qx = (R[0, 1] + R[1, 0]) / S_val
        qy = 0.25 * S_val
        qz = (R[1, 2] + R[2, 1]) / S_val
    else:
        S_val = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S_val
        qx = (R[0, 2] + R[2, 0]) / S_val
        qy = (R[1, 2] + R[2, 1]) / S_val
        qz = 0.25 * S_val

    q_e = np.array([qw, qx, qy, qz], dtype=np.float32)
    V_vol = np.array([4.0 / 3.0 * np.pi * r[0] * r[1] * r[2]], dtype=np.float32)

    return (
        torch.from_numpy(c),
        torch.from_numpy(r),
        torch.from_numpy(q_e),
        torch.from_numpy(V_vol),
        torch.from_numpy(J_arm.astype(np.float32)),
    )


def main(repo_id="gr1_pickup_grasp"):
    lerobot_home = os.environ.get("LEROBO_HOME")
    if lerobot_home:
        local_path = Path(lerobot_home) / repo_id
        if local_path.exists():
            dataset = LeRobotDataset(repo_id, root=str(local_path))
        else:
            dataset = LeRobotDataset(repo_id)
    else:
        dataset = LeRobotDataset(repo_id)
    dataset_path = Path(dataset.root)

    # Initialize MuJoCo once for caching
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    RIGHT_ARM_JOINTS = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
    ]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R_index_tip_link")
    dof_indices = []
    for name in RIGHT_ARM_JOINTS:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id != -1:
            dof_indices.append(model.jnt_dofadr[j_id])

    cache_dir = dataset_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    views = ["world_center", "world_left", "world_right", "world_top", "world_wrist"]
    total_episodes = dataset.num_episodes

    print(
        "🚀 Compiling High-Speed Direct Fused Disk Cache with Reachability Map Priors..."
    )
    print(f"🎬 Target: {total_episodes} Episodes (Center, Left, Right, Top, Wrist)")

    for ep in tqdm(range(total_episodes), desc="Caching Episodes"):
        ep_meta = dataset.meta.episodes[ep]
        c_idx = ep_meta["data/chunk_index"]
        f_idx = ep_meta["data/file_index"]

        # 1. Load Parquet for actions and states
        parquet_path = dataset_path / f"data/chunk-{c_idx:03d}/file-{f_idx:03d}.parquet"
        if not parquet_path.exists():
            print(f"⚠️ Parquet missing for Episode {ep}: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        state_cols = [c for c in df.columns if c.startswith("observation.state")]
        action_cols = [c for c in df.columns if c.startswith("action")]

        if len(state_cols) == 1:
            state_tensor = torch.from_numpy(np.stack(df[state_cols[0]].values)).float()
        else:
            state_tensor = torch.from_numpy(df[state_cols].values).float()

        if len(action_cols) == 1:
            action_tensor = torch.from_numpy(
                np.stack(df[action_cols[0]].values)
            ).float()
        else:
            action_tensor = torch.from_numpy(df[action_cols].values).float()

        # Compute reachability ellipsoids for all 32 steps in the episode
        c_list, r_list, qe_list, v_list, J_list = [], [], [], [], []
        for t in range(32):
            c, r, q_e, V_vol, J_arm = compute_ellipsoid_analytically(
                model, data, dof_indices, ee_id, state_tensor[t]
            )
            c_list.append(c)
            r_list.append(r)
            qe_list.append(q_e)
            v_list.append(V_vol)
            J_list.append(J_arm)

        ellipsoid_center = torch.stack(c_list, dim=0)
        ellipsoid_radii = torch.stack(r_list, dim=0)
        ellipsoid_quat = torch.stack(qe_list, dim=0)
        ellipsoid_volume = torch.stack(v_list, dim=0)
        ellipsoid_jacobian = torch.stack(J_list, dim=0)

        # Load video frames
        episode_pixels = []
        for view in views:
            skel_path = (
                dataset_path
                / f"videos/observation.images.{view}_tiled/chunk-{c_idx:03d}/file-{f_idx:03d}.mp4"
            )

            if not skel_path.exists():
                continue

            cap_skel = cv2.VideoCapture(str(skel_path))
            view_frames = []
            for frame_idx in range(32):
                ret_skel, frame_tiled = cap_skel.read()

                if not ret_skel:
                    rgb_224 = np.zeros((224, 224, 3), dtype=np.uint8)
                    skel_224 = np.zeros((224, 224), dtype=np.uint8)
                else:
                    h_tiled, w_tiled, _ = frame_tiled.shape
                    if w_tiled == 960:
                        frame_rgb = frame_tiled[:, :480]
                        frame_skel = cv2.cvtColor(
                            frame_tiled[:, 480:], cv2.COLOR_BGR2GRAY
                        )
                    elif w_tiled == 448:
                        frame_rgb = frame_tiled[:, :224]
                        frame_skel = cv2.cvtColor(
                            frame_tiled[:, 224:], cv2.COLOR_BGR2GRAY
                        )
                    else:
                        mid = w_tiled // 2
                        frame_rgb = frame_tiled[:, :mid]
                        frame_skel = cv2.cvtColor(
                            frame_tiled[:, mid:], cv2.COLOR_BGR2GRAY
                        )

                    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                    rgb_224 = cv2.resize(
                        frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR
                    )
                    skel_224 = cv2.resize(
                        frame_skel, (224, 224), interpolation=cv2.INTER_AREA
                    )

                fused = np.zeros((4, 224, 224), dtype=np.uint8)
                fused[:3] = rgb_224.transpose(2, 0, 1)
                fused[3] = skel_224
                view_frames.append(torch.from_numpy(fused))

            cap_skel.release()
            view_tensor = torch.stack(view_frames, dim=0)
            episode_pixels.append(view_tensor)

        stacked_pixels = torch.stack(episode_pixels, dim=1)

        dino_pt_path = (
            dataset_path / f"cache_dino/chunk-{c_idx:03d}/file-{f_idx:03d}_dino.pt"
        )
        if dino_pt_path.exists():
            dino_waypoints = validate_dino_waypoints(
                torch.load(dino_pt_path, map_location="cpu")
            )
        else:
            dino_waypoints = zeros_dino_waypoints()

        # Pack ellipsoid parameters into serialized dictionary
        packaged_data = {
            "pixels": stacked_pixels,
            "state": state_tensor,
            "action": action_tensor,
            "dino_waypoints": dino_waypoints,
            "ellipsoid_center": ellipsoid_center,
            "ellipsoid_radii": ellipsoid_radii,
            "ellipsoid_quat": ellipsoid_quat,
            "ellipsoid_volume": ellipsoid_volume,
            "ellipsoid_jacobian": ellipsoid_jacobian,
        }

        out_path = cache_dir / f"episode_{ep:03d}_fused.pt"
        buffer = io.BytesIO()
        torch.save(packaged_data, buffer)
        compressed = lz4.frame.compress(buffer.getvalue())
        with open(out_path, "wb") as f:
            f.write(compressed)

    print(
        f"🎉 Pre-compiled cache with ellipsoids successfully generated inside: {cache_dir}"
    )


if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "gr1_pickup_grasp"
    main(repo_id)
