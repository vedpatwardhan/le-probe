import pytest
import numpy as np


def test_mu_to_numpy_safe_extraction(sim):
    """
    Ensures that MuJoCo state extraction is safely moved to NumPy for ZMQ.
    """
    # 1. Test get_state_32 (Main I/O boundary)
    state = sim.get_state_32()
    assert isinstance(state, np.ndarray), f"Expected NumPy array, got {type(state)}"
    assert state.dtype == np.float32
    assert state.shape == (32,)

    # 2. Test solve_ik extraction
    pos = (0.45, 0.0, 1.0)
    quat = (0.707, 0, 0.707, 0)
    target_q = sim.solve_ik(pos, quat)

    assert isinstance(target_q, np.ndarray)
    assert target_q.shape == (sim.model.nq,)


def test_waist_randomization_boundaries(sim):
    """
    Assures that the randomized waist-pitch is updated during reset_env.
    """
    idx_waist = 30  # Protocol Index for waist_pitch_joint
    j_id = sim.protocol_joint_ids[idx_waist]
    q_idx = sim.model.jnt_qposadr[j_id]

    # Run 10 trials to check randomization presence
    initial_q = sim.data.qpos.copy()

    any_movement = False
    for _ in range(10):
        sim.reset_env()
        actual_rad = sim.data.qpos[q_idx]
        if abs(actual_rad - initial_q[q_idx]) > 0.001:
            any_movement = True
            break

    assert any_movement, "Waist pitch failed to randomize during reset_env!"


def test_rendering_sanity(sim):
    """
    Verifies that the MuJoCo renderer produces valid image buffers.
    """
    sim.reset_env()
    # Capture from first camera
    cam_name = sim.cam_names[0]
    sim.renderer.update_scene(sim.data, camera=cam_name)
    rgb = sim.renderer.render()

    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (sim.res[1], sim.res[0], 3)
    assert rgb.dtype == np.uint8
    assert np.any(rgb > 0), "Rendered image is completely black!"
