import pytest
import os
import sys

# Ensure gr1_gr00t is in path if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation_base import GR1MuJoCoBase


@pytest.fixture(scope="session")
def sim():
    """
    Provides a singleton instance of the GR1 MuJoCo Base for tests.
    """
    # Initialize the base simulation (Physical layer only)
    simulation = GR1MuJoCoBase()
    return simulation
