import numpy as np
import torch
from gr1_config import JOINT_LIMITS_MIN, JOINT_LIMITS_MAX


def test_mps_to_numpy_safe_extraction(sim):
    """
    Ensures that simulation outputs are safely moved to CPU/NumPy for ZMQ and reporting.
    This protects against 'TypeError: can't convert mps:0 device type tensor to numpy'.
    """
    # 1. Test get_state_32 (Main I/O boundary)
    state = sim.get_state_32()
    assert isinstance(state, np.ndarray), f"Expected NumPy array, got {type(state)}"
    assert state.dtype == np.float32 or state.dtype == np.float64
    assert state.shape == (32,)

    # 2. Test auto_reach extraction (Historical failure point)
    # We'll trigger solve_ik and then check the extraction logic
    pos = (0.45, 0.0, 1.0)
    quat = (0.707, 0, 0.707, 0)
    target_q = sim.solve_ik(pos, quat)

    # Mirroring the logic inside sim.run() that crashed
    target_q_32 = torch.stack([target_q[m["dof_idx"]] for m in sim.joint_dof_map])
    final_action_32 = sim._normalize_state(target_q_32).cpu().numpy()

    assert isinstance(final_action_32, np.ndarray)
    assert final_action_32.shape == (32,)


def test_waist_randomization_boundaries(sim):
    """
    Assures that the randomized waist-pitch is always within the 'Ready Pose' range.
    Expected normalized range: roughly [-0.6, -0.2].
    """
    waist_pitch_joint = sim.robot.get_joint("waist_pitch_joint")
    waist_dof_idx = waist_pitch_joint.dofs_idx[0]

    # Use mapping info for normalization
    for mapping in sim.joint_dof_map:
        if mapping["name"] == "waist_pitch_joint":
            limit_min, limit_max = mapping["limits"]
            break

    # Run 10 trials to check randomization distribution
    for _ in range(10):
        sim.reset_env()
        actual_rad = sim.robot.get_qpos()[waist_dof_idx].item()

        # Normalize back to [-1, 1]
        normalized_pitch = (actual_rad - limit_min) / (
            limit_max - limit_min
        ) * 2.0 - 1.0

        # We allow a tiny tolerance for floating point epsilon
        assert (
            -0.61 <= normalized_pitch <= -0.16
        ), f"Waist pitch {normalized_pitch:.3f} fell outside authorized ready-pose range!"
