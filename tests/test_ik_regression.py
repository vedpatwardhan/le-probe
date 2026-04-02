import pytest
import numpy as np
import os
import sys

# Ensure gr00t-gr1-pickup is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_base import GR1MuJoCoBase


@pytest.fixture
def sim():
    return GR1MuJoCoBase()


def test_ik_parameter_validation_crash_prevention(sim):
    """
    Ensures that passing a length-3 vector (like a position) to the 'quat'
    parameter triggers a helpful ValueError instead of a cryptic Mink exception.
    """
    pos = [0.4, 0.0, 0.8]
    bad_quat = [1.0, 0.0, 0.0]  # Length 3

    with pytest.raises(ValueError) as excinfo:
        sim.solve_ik(pos, bad_quat)

    assert "quat must be length 4" in str(excinfo.value)
    assert "got 3" in str(excinfo.value)


def test_ik_optional_arguments(sim):
    """
    Tests that solve_ik still works with just (pos, quat) due to default arguments.
    """
    pos = [0.4, 0.0, 0.8]
    quat = [1, 0, 0, 0]

    q_res = sim.solve_ik(pos, quat)
    assert q_res is not None
    assert len(q_res) == sim.model.nq


def test_ik_full_arguments_reordered(sim):
    """
    Tests the standardized 4-argument call signature.
    """
    pos_w = [0.4, 0.0, 0.8]
    pos_i = [0.4, 0.0, 0.85]
    pos_t = [0.4, 0.0, 0.85]
    quat = [1, 0, 0, 0]

    # Standard order: (pos_wrist, quat, pos_index, pos_thumb)
    q_res = sim.solve_ik(pos_w, quat, pos_i, pos_t)
    assert q_res is not None
