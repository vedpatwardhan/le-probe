# Le-Probe: Probing Embodied Intelligence

<img src="banner.png" width="100%" height="350" style="object-fit: cover; border-radius: 12px; margin-bottom: 20px;">

> **Probing the behavior of World Models (LeWM) in high-DoF tasks.**

Le-Probe is a research framework designed to analyze and compare the performance of **World Models** versus traditional **Vision-Language-Action (VLA)** policies in complex, multi-phase robotic tasks. 

Our primary goal is to understand how latent imagination can handle high-dimensional control (32+ DoF) in tasks that require non-monotonic progress (e.g., moving away from a goal to enable a better grasp).

## 🚀 Repository Structure

The project is organized into functional modules:

- [**`dataset/`**](./dataset): Teleoperation and high-fidelity data collection.
- [**`vla/`**](./vla): End-to-end foundation model inference (GR00T).
- [**`lewm/`**](./lewm): World model training, imagination, and MPC planning.
- [**`interpretability/`**](./interpretability): Probing and visualization tools (Coming Soon).
- [**`scripts/`**](./scripts): Maintenance and migration utilities.

## 🔬 Core Mission

Current VLAs often struggle with long-horizon tasks that require intermediate sub-goals. **Le-Probe** investigates whether **LeWM** (LeRobot World Model) can provide superior robustness by:
1. **Latent Imagination**: Visualizing future states before execution.
2. **Oracle MPC**: Searching for optimal trajectories using a learned reward manifold.
3. **Behavioral Probing**: Systematic analysis of failure modes in high-DoF environments.

## 🛠 Getting Started

### 1. Installation
```bash
git clone --recursive https://github.com/vedpatwardhan/le-probe.git
cd le-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Quick Start (Teleop)
```bash
# Start simulation
.venv/bin/python dataset/simulation_teleop.py

# Launch Dashboard
streamlit run dataset/teleop_ui.py
```

---
*Developed by Ved Patwardhan.*
