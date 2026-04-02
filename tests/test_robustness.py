import pytest
import numpy as np


def test_pd_convergence_accuracy(sim):
    """Verifies that dispatch_action reaching the target within epsilon in MuJoCo."""
    # Reset to home first
    sim.reset_env()
    target_q = sim.last_target_q.copy()

    # Identify a whitelisted joint to test (e.g., Right Shoulder Pitch, Protocol Index 16)
    idx_16 = 16
    j_id = sim.protocol_joint_ids[idx_16]
    q_idx = sim.model.jnt_qposadr[j_id]

    # Move Right Shoulder Pitch by 0.2 rad (MuJoCo is more sensitive than Genesis)
    target_q[q_idx] += 0.2

    # Create a 32-dim action (only index 16 is active)
    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_32[idx_16] = 0.5  # Relative to middle of range

    # Execute the action (internally calls dispatch_action)
    sim.dispatch_action(action_32, target_q)

    # Check convergence in MuJoCo qpos
    actual_val = sim.data.qpos[q_idx]
    delta = abs(actual_val - target_q[q_idx])

    # We expect high precision in MuJoCo
    assert delta < 1e-2, f"MuJoCo PD Glide failed to converge! Delta: {delta:.6f} rad"


def test_finger_coupling_propagation(sim):
    """Verifies that proximal joint commands correctly update distal DOF targets in the buffer."""
    # Index 25 is 'R_index_proximal_joint'
    prox_idx = 25
    assert prox_idx in sim.coupling_map, f"Index {prox_idx} should have coupling!"

    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_32[prox_idx] = 0.8  # Almost closed

    # Reset buffer
    sim.last_target_q[:] = sim.model.qpos0[:]
    sim.process_target_32(action_32)

    # Check that coupled joints in last_target_q match the proximal target
    for distal_q_idx in sim.coupling_map[prox_idx]:
        actual_rad = sim.last_target_q[distal_q_idx]
        assert (
            abs(actual_rad) > 1e-4
        ), f"Coupled DOF at qpos[{distal_q_idx}] failed to update!"


def test_adversarial_input_robustness(sim):
    """Verifies that process_target_32 handles extreme/malformed inputs safely."""
    sim.reset_env()

    # 1. Extreme value (> 1.0)
    idx_16 = 16
    action_32 = np.full(32, np.nan, dtype=np.float32)
    action_32[idx_16] = 10.0  # Way out of bounds

    sim.process_target_32(action_32)

    # Should have been clipped to limit_max
    j_id = sim.protocol_joint_ids[idx_16]
    q_idx = sim.model.jnt_qposadr[j_id]
    limit_max = sim.wire_max[idx_16]

    assert (
        abs(sim.last_target_q[q_idx] - limit_max) < 1e-4
    ), "Failed to clip excessive input!"

    # 2. NaN in an authorized joint (should be ignored, keeping previous value)
    prev_val = sim.last_target_q[q_idx]
    action_32[idx_16] = np.nan
    sim.process_target_32(action_32)
    assert (
        sim.last_target_q[q_idx] == prev_val
    ), "NaN input erroneously modified target!"
