import pytest
import numpy as np


def test_32dim_action_mapping(sim):
    """Verifies that a 32-dim action vector correctly maps to the MuJoCo ctrl buffer."""
    sim.reset_env()

    # 1. Create a sparse action at index 16 (Right Shoulder Pitch)
    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_16_val = 0.5  # Mid-point of positive range
    action_32[16] = action_16_val

    # 2. Process the action (updates last_target_q)
    sim.process_target_32(action_32)

    # Check intermediate buffer (last_target_q)
    j_id = sim.protocol_joint_ids[16]
    q_idx = sim.model.jnt_qposadr[j_id]
    expected_rad = (action_16_val + 1.0) / 2.0 * (
        sim.wire_max[16] - sim.wire_min[16]
    ) + sim.wire_min[16]
    assert abs(sim.last_target_q[q_idx] - expected_rad) < 1e-4

    # 3. Dispatch the action (updates ctrl buffer)
    sim.dispatch_action(action_32, sim.last_target_q)

    # Actuators in our XML map 1:1 to joints for the enabled 32 DOFs
    # We find the actuator corresponding to joint ID
    found_actuator = False
    for a_id in range(sim.model.nu):
        if sim.model.actuator_trnid[a_id, 0] == j_id:
            assert abs(sim.data.ctrl[a_id] - expected_rad) < 1e-4
            found_actuator = True
            break
    assert found_actuator, "Could not find actuator for whitelisted protocol joint!"


def test_protocol_normalization_limits(sim):
    """Verifies that -1.0/1.0 actions correctly map to absolute min/max limits."""
    # Test Index 16 (Right Shoulder Pitch) - usually authorized
    idx_target = 16
    j_id = sim.protocol_joint_ids[idx_target]
    if j_id == -1 or sim.v_allowed_mask[idx_target] < 0.5:
        pytest.skip(f"Protocol Index {idx_target} not in current whitelist.")

    q_idx = sim.model.jnt_qposadr[j_id]
    action_32 = np.full(32, np.nan, dtype=np.float32)

    # Test Min
    action_32[idx_target] = -1.0
    sim.process_target_32(action_32)
    assert abs(sim.last_target_q[q_idx] - sim.wire_min[idx_target]) < 1e-4

    # Test Max
    action_32[idx_target] = 1.0
    sim.process_target_32(action_32)
    assert abs(sim.last_target_q[q_idx] - sim.wire_max[idx_target]) < 1e-4


def test_state_reconstruction_accuracy(sim):
    """Verifies that get_state_32 correctly reconstructs the 32-dim vector from MuJoCo state."""
    sim.reset_env()

    # Set a known state for index 16
    idx_16 = 16
    j_id = sim.protocol_joint_ids[idx_16]
    q_idx = sim.model.jnt_qposadr[j_id]

    test_val = (sim.wire_max[idx_16] + sim.wire_min[idx_16]) / 2.0 + 0.1
    sim.data.qpos[q_idx] = test_val

    state_32 = sim.get_state_32()

    # Expected normalized value
    rng = sim.wire_max[idx_16] - sim.wire_min[idx_16]
    expected_norm = 2.0 * (test_val - sim.wire_min[idx_16]) / rng - 1.0

    assert abs(state_32[idx_16] - expected_norm) < 1e-4


def test_finger_coupling_logic(sim):
    """Verifies that moving a proximal finger joint automatically updates its distal siblings."""
    # Find a coupled joint (e.g., R_thumb_proximal_yaw_joint at index 23)
    prox_idx = 23
    if prox_idx not in sim.coupling_map:
        pytest.skip(f"Index {prox_idx} should have coupling in this test config.")

    # Apply action
    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_32[prox_idx] = 1.0  # Fully extended
    sim.process_target_32(action_32)

    # Verify all coupled DOFs moved to matching values in last_target_q
    prox_j_id = sim.protocol_joint_ids[prox_idx]
    prox_q_idx = sim.model.jnt_qposadr[prox_j_id]
    target_rad_val = sim.last_target_q[prox_q_idx]

    for distal_q_idx in sim.coupling_map[prox_idx]:
        coupled_val = sim.last_target_q[distal_q_idx]
        assert abs(coupled_val - target_rad_val) < 1e-6
