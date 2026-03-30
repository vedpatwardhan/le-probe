import numpy as np
import torch
from gr1_config import COMPACT_WIRE_JOINTS


def test_32_dof_mapping_consistency(sim):
    """
    Exhaustively verifies that every 32-DOF input maps to its correct physical joint.
    """
    # 1. Reset target q
    sim.last_target_q = sim.home_q.clone()

    # 2. Iterate through all 32 indices to verify the entire protocol
    for target_idx in range(32):
        mapping = sim.joint_dof_map[target_idx]
        joint_name = COMPACT_WIRE_JOINTS[target_idx]

        # SOURCE OF TRUTH: Query the robot directly by name, bypassing the mapping
        ground_truth_dof = sim.robot.get_joint(joint_name).dofs_idx[0]

        # 3. Create a isolated packet for this joint
        action_32 = np.full(32, np.nan, dtype=np.float32)
        action_32[target_idx] = 0.8

        # 4. Process (only if authorized)
        sim.process_target_32(action_32)

        if target_idx in sim.allowed_32_indices:
            # Verify the target was set correctly
            limit_min, limit_max = mapping["limits"]
            expected_rad = (0.8 + 1.0) / 2.0 * (limit_max - limit_min) + limit_min
            actual_rad = sim.last_target_q[ground_truth_dof].item()
            assert (
                abs(actual_rad - expected_rad) < 1e-4
            ), f"Mapping error at index {target_idx} ({joint_name})!"
        else:
            # Verify it was rejected
            actual_rad = sim.last_target_q[ground_truth_dof].item()
            home_rad = sim.home_q[ground_truth_dof].item()
            assert (
                abs(actual_rad - home_rad) < 1e-6
            ), f"Unauthorized joint {joint_name} bypassed the gain at index {target_idx}!"


def test_finger_coupling_logic(sim):
    """
    Verifies that moving a proximal finger joint automatically updates its distal siblings.
    """
    # 1. Find a coupled joint (e.g., R_index_proximal_joint at index 25)
    target_idx = 25
    mapping = sim.joint_dof_map[target_idx]
    assert "proximal" in mapping["name"].lower()
    assert (
        len(mapping["coupled"]) > 0
    ), f"Joint {mapping['name']} should have coupled DOFs!"

    # 2. Reset
    sim.last_target_q = sim.home_q.clone()

    # 3. Apply action
    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_32[target_idx] = 1.0  # Fully extended
    sim.process_target_32(action_32)

    # 4. Verify all coupled DOFs moved to matching values
    target_val = sim.last_target_q[mapping["dof_idx"]].item()
    for coupled_dof_idx in mapping["coupled"]:
        coupled_val = sim.last_target_q[coupled_dof_idx].item()
        assert (
            abs(coupled_val - target_val) < 1e-6
        ), f"Coupled DOF {coupled_dof_idx} failed to sync with proximal joint!"


def test_unauthorized_joint_rejection(sim):
    """
    Ensures that process_target_32 explicitly ignores joints NOT in the whitelist.
    """
    # 1. Pick an unauthorized index (e.g., index 0: left_shoulder_pitch_joint)
    unauth_idx = 0
    assert unauth_idx not in sim.allowed_32_indices

    # 2. Reset target q
    sim.last_target_q = sim.home_q.clone()
    original_val = sim.last_target_q[sim.joint_dof_map[unauth_idx]["dof_idx"]].item()

    # 3. Attempt to move it
    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_32[unauth_idx] = 1.0
    sim.process_target_32(action_32)

    # 4. Verify it didn't move
    final_val = sim.last_target_q[sim.joint_dof_map[unauth_idx]["dof_idx"]].item()
    assert (
        final_val == original_val
    ), "Unauthorized joint was able to bypass the intake gate!"
