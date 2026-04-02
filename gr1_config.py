import numpy as np
import xml.etree.ElementTree as ET
import os

# -----------------------------------------------------------------------------
# GR1 Robot Configuration
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# JOINT NAMES (Canonical Protocol - Updated for Fourier URDF)
# Model Expects LEFT side first (GR00T-N1.5 Standard)
# -----------------------------------------------------------------------------
L_ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
]
R_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]
L_HAND_NAMES = [
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
]
R_HAND_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]
HEAD_NAMES = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
WAIST_NAMES = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]

# DEFINITIVE WIRE PROTOCOL (ZMQ Communication)
# 32-dim protocol aligned with the model's action head
COMPACT_WIRE_JOINTS = (
    L_ARM_NAMES  # 0-6
    + L_HAND_NAMES  # 7-12
    + HEAD_NAMES  # 13-15
    + R_ARM_NAMES  # 16-22
    + R_HAND_NAMES  # 23-28
    + WAIST_NAMES  # 29-31
)
FROZEN_JOINTS = {
    "head_pitch_joint": 0.0,
    "head_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
}

CAMERA_ATTACH_LINK = "right_hand_pitch_link"

# Central Paths
XML_PATH = "/content/sim_assets/gr1t1_fourier_hand_6dof.xml"
if not os.path.exists(XML_PATH):
    XML_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/gr1_gr00t/sim_assets/gr1t1_fourier_hand_6dof.xml"

SCENE_PATH = "/content/sim_assets/scene_gr1_pickup.xml"
if not os.path.exists(SCENE_PATH):
    SCENE_PATH = "/Users/vedpatwardhan/Desktop/cortex-os/gr1_gr00t/sim_assets/scene_gr1_pickup.xml"


# -----------------------------------------------------------------------------
# PHYSICAL BONES (Dynamic Sync 🔗)
# -----------------------------------------------------------------------------
def get_limits(xml_path, joint_names):
    """Scans robotic asset files (URDF/XML) for joint limit bounds."""
    if not os.path.exists(xml_path):
        return np.zeros(len(joint_names), dtype=np.float32), np.zeros(
            len(joint_names), dtype=np.float32
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_db = {}
    for j in root.iter("joint"):
        name = j.get("name")
        if name is None:
            continue

        # Try URDF style: <joint><limit lower="..." upper="..." /></joint>
        limit_tag = j.find("limit")
        if limit_tag is not None:
            lower = float(limit_tag.get("lower", 0.0))
            upper = float(limit_tag.get("upper", 0.0))
            joint_db[name] = (lower, upper)
            continue

        # Try MuJoCo style: <joint range="min max" />
        range_attr = j.get("range")
        if range_attr is not None:
            parts = range_attr.split()
            if len(parts) == 2:
                lower = float(parts[0])
                upper = float(parts[1])
                joint_db[name] = (lower, upper)
                continue

        # Fallback
        joint_db[name] = (0.0, 0.0)

    mins = [
        joint_db.get(j, (0.0, 0.0))[0] if j not in FROZEN_JOINTS else FROZEN_JOINTS[j]
        for j in joint_names
    ]
    maxs = [
        joint_db.get(j, (0.0, 0.0))[1] if j not in FROZEN_JOINTS else FROZEN_JOINTS[j]
        for j in joint_names
    ]
    return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)


JOINT_LIMITS_MIN, JOINT_LIMITS_MAX = get_limits(XML_PATH, COMPACT_WIRE_JOINTS)

# -----------------------------------------------------------------------------
# DIAGNOSTIC REPORT 📋
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- GR1 Protocol Config (XML: {os.path.basename(XML_PATH)}) ---")
    print(f"Camera Mount: {CAMERA_ATTACH_LINK}")
    print(f"{'Idx':<4} {'Joint Name':<40} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for i, name in enumerate(COMPACT_WIRE_JOINTS):
        print(
            f"{i:<4} {name:<40} {JOINT_LIMITS_MIN[i]:<10.3f} {JOINT_LIMITS_MAX[i]:<10.3f}"
        )
    print("-" * 70)
    print(f"Total DOFs: {len(COMPACT_WIRE_JOINTS)}\n")
