import pytest
from gr1_gr00t.simulation import GR1Simulation


@pytest.fixture(scope="session")
def sim():
    """
    Provides a singleton, headless instance of the GR1 Simulation for tests.
    test_mode=True ensures no ZMQ or Rerun overhead.
    """
    # Initialize in test mode (no ZMQ, no Rerun)
    simulation = GR1Simulation(test_mode=True)
    return simulation
