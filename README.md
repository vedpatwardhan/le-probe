# Le-Probe: Probing Embodied World Models

<img src="banner.png" width="100%" height="350" style="object-fit: cover; border-radius: 12px; margin-bottom: 20px;">

Le-Probe is a research framework designed to analyze and compare **LeRobot World Models (LeWM)** against traditional **Vision-Language-Action (VLA)** policies like GR00T-N1. 

Our investigation focuses on high-DoF (32+) manipulation tasks that require multi-phase coordination, specifically comparing two distinct behavioral strategies: **Grasp** and **Cup**.

## 🚀 Repository Structure

- [**`dataset/`**](./dataset): Teleoperation and high-fidelity data collection (32-frame episodes).
- [**`vla/`**](./vla): GR00T-N1 baselines. Successfully demonstrates both Grasp and Cup behaviors.
- [**`lewm/`**](./lewm): World model training and Oracle MPC. Currently struggles with latent discriminability.
- [**`interpretability/`**](./interpretability): The "Search for the Why"—mechanistic analysis of LeWM failure modes.
- [**`scripts/`**](./scripts): Maintenance, dataset compression, and reward calibration tools.

## 🔬 Core Mission: VLA vs. LeWM

The project was born from a comparative study of two approaches to the same task: picking up a red cube.

### 1. VLA Success (GR00T-N1)
We successfully trained GR00T-N1 to imitate two different movement styles. Despite early protocol mismatches, the "Parity Refactor" stabilized the inference stack, allowing GR00T to execute both behaviors reliably.

| Grasp Movement | Cup Movement |
| :---: | :---: |
| <video width="100%" controls><source src="assets/vla_grasp.mp4" type="video/mp4"></video> | <video width="100%" controls><source src="assets/vla_cup.mp4" type="video/mp4"></video> |

### 2. LeWM Challenges (The Discriminability Gap)
LeWM, despite training with a large softrank, failed to sufficiently discriminate the goal state from non-goal states in the latent space. 

| LeWM MPC Inference (Clipped) |
| :---: |
| <video width="100%" controls><source src="assets/lewm_grasp.mp4" type="video/mp4"></video> |

*   **Reward Head Intervention**: To fix this, we trained an auxiliary reward head on snapshot data. While reward prediction is now accurate, the MPC solver often fails to find trajectories as smooth or effective as the VLA baseline.
*   **Current Status**: We are currently focused on why "good imagination" in the JEPA architecture does not always translate to "good action" in high-DoF control.

## 🛠 Getting Started

```bash
# 1. Install
git clone --recursive https://github.com/vedpatwardhan/le-probe.git
cd le-probe && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run Teleop (Data Collection)
.venv/bin/python dataset/simulation_teleop.py
streamlit run dataset/teleop_ui.py
```

---
*Developed by Ved Patwardhan.*
