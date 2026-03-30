import pytest
import numpy as np
import torch


def test_pd_convergence_accuracy(sim):
    """Verifies that dispatch_action (PD glide) actually reaches the target within epsilon."""
    target_q = sim.home_q.clone()
    # Move Right Shoulder Pitch by 0.5 rad
    mapping = sim.joint_dof_map[16]
    dof_idx = mapping["dof_idx"]
    target_q[dof_idx] += 0.5

    start_q = sim.robot.get_qpos().clone()
    # Execute a unified glide (Uniform: 200 steps)
    sim.dispatch_action(np.full(32, np.nan), target_q, start_q=start_q)

    actual_q = sim.robot.get_qpos()
    delta = abs(actual_q[dof_idx] - target_q[dof_idx]).item()

    # We expect high precision (~2.8 degrees)
    assert delta < 5e-2, f"PD Glide failed to converge! Delta: {delta:.6f} rad"


def test_finger_coupling_propagation(sim):
    """Verifies that proximal joint commands correctly update distal DOF targets."""
    # Index 25 is 'R_index_proximal_joint'
    prox_idx = 25
    mapping = sim.joint_dof_map[prox_idx]
    assert (
        len(mapping["coupled"]) > 0
    ), f"Index {prox_idx} ({mapping['name']}) has no coupled joints!"

    action_32 = np.full(32, np.nan)
    action_32[prox_idx] = 0.5  # Partially closed

    sim.process_target_32(action_32)

    # Check that coupled joints in last_target_q match the proximal target
    # 0.5 maps to a non-zero value, so we just verify they moved
    for c_idx in mapping["coupled"]:
        actual_rad = sim.last_target_q[c_idx]
        assert (
            abs(actual_rad) > 1e-4
        ), f"Coupled DOF {c_idx} failed to update! Actual: {actual_rad}"


def test_adversarial_input_robustness(sim):
    """Verifies that process_target_32 handles extreme/malformed inputs safely."""
    # 1. Extreme value (> 1.0)
    action_32 = np.full(32, np.nan)
    action_32[16] = 10.0  # Way out of bounds

    sim.process_target_32(action_32)

    # Should have been clipped to 1.0
    mapping = sim.joint_dof_map[16]
    limit_max = mapping["limits"][1]
    assert (
        abs(sim.last_target_q[mapping["dof_idx"]] - limit_max) < 1e-4
    ), "Failed to clip excessive input!"

    # 2. NaN in an authorized joint (should be ignored)
    prev_val = sim.last_target_q[mapping["dof_idx"]]
    action_32[16] = np.nan
    sim.process_target_32(action_32)
    assert (
        sim.last_target_q[mapping["dof_idx"]] == prev_val
    ), "NaN input erroneously modified target!"
