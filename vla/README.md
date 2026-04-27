# 🧠 Vision-Language-Action (VLA) Inference

> End-to-end autonomous control using state-of-the-art foundation models.

This module hosts the inference servers and autonomous drivers for VLA models like **GR00T**. It bridges the gap between high-level reasoning and low-level physical control.

## 🛠 Key Components

- **`simulation_vla.py`**: The autonomous mission driver that captures observations and executes VLA actions in the simulation.
- **`gr00t_server.py`**: A high-performance ZMQ server for hosting the GR00T policy (typically runs on Colab/Remote GPU).
- **`GR00T_N1_BC.ipynb`**: Training notebook for Behavioral Cloning with GR00T-N1.
- **`GR00T_N1_E2E.ipynb`**: End-to-end evaluation pipeline.

## 🚀 Usage

### Running an Autonomous Mission
1. Ensure the `gr00t_server.py` is running on your GPU instance.
2. Launch the simulation driver:
   ```bash
   .venv/bin/python vla/simulation_vla.py --host <server_ip> --instruction "Pick up the red cube"
   ```

---
*Part of the [Le-Probe](..) project.*
