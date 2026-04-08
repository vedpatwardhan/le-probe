import pytest
import numpy as np
from gr1_config import COMPACT_WIRE_JOINTS


def test_protocol_completeness(sim):
    """Ensure that the protocol defines exactly 32 DOFs and all are resolved."""
    # 1. Dimension Check
    assert len(COMPACT_WIRE_JOINTS) == 32, "Protocol must be exactly 32 DOFs"
    assert len(sim.protocol_joint_ids) == 32, "Internal joint mapping list mismatch"
    assert sim.v_allowed_mask.shape == (32,), "Allowed mask dimension mismatch"

    # 2. Key Joint Resolution
    # Indices 0-6 (Left Arm) and 16-22 (Right Arm) are non-negotiable
    essential_indices = [0, 3, 16, 19]  # Shoulders and Elbows
    for idx in essential_indices:
        j_id = sim.protocol_joint_ids[idx]
        assert (
            j_id != -1
        ), f"Essential joint at index {idx} ({COMPACT_WIRE_JOINTS[idx]}) failed to resolve!"


def test_joint_limits_broadcast(sim):
    """Verify that wire_min/max are populated from the config/XML."""
    assert sim.wire_min.shape == (32,)
    assert sim.wire_max.shape == (32,)

    # Ensure they aren't uninitialized/all zero
    # (Checking indices 0-6 which always have limits in Fourier URDF)
    assert np.any(sim.wire_min[0:7] != 0.0)
    assert np.any(sim.wire_max[0:7] != 0.0)

    # Range check
    assert np.all(
        sim.wire_max >= sim.wire_min
    ), "Max limit cannot be less than min limit"


def test_coupling_map_integrity(sim):
    """Ensure coupling map indices point to valid qpos addresses."""
    # Coupling map should handle finger distal mimicing
    # Left hand starts at index 7, Right hand at index 23
    hand_indices = list(range(7, 13)) + list(range(23, 29))

    coupling_found = False
    for prox_idx, distal_q_indices in sim.coupling_map.items():
        assert 0 <= prox_idx < 32, "Source index out of protocol range"
        assert (
            prox_idx in hand_indices
        ), "Coupling should only apply to hand/finger proximal joints"

        for q_idx in distal_q_indices:
            assert 0 <= q_idx < sim.model.nq, "Coupled qpos index out of model range"
            coupling_found = True

    assert coupling_found, "Config reported no coupled joints (fingers likely unmapped)"


def test_allowed_mask_coverage(sim):
    """Verify that the v_allowed_mask actually covers relevant joints."""
    # At least the arms should be allowed for any policy to work
    arm_mask = sim.v_allowed_mask[16:23]  # Right Arm
    assert np.all(
        arm_mask == 1.0
    ), "Active arm joints must be allowed in v_allowed_mask"
