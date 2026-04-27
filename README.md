# Le-Probe: Probing LeWM

<div align="center">
  <img src="banner.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
</div>

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

### 1. Target Behaviors (Ground Truth)
We maintained two high-quality datasets representing different manipulation priors.

<div align="center">
  <table>
    <tr>
      <th>Dataset: Grasp Pattern</th>
      <th>Dataset: Cup Pattern</th>
    </tr>
    <tr>
      <td><video src="https://github.com/vedpatwardhan/le-probe/raw/main/assets/dataset_grasp.mp4?raw=true" controls width="100%"></video></td>
      <td><video src="https://github.com/vedpatwardhan/le-probe/raw/main/assets/dataset_cup.mp4?raw=true" controls width="100%"></video></td>
    </tr>
  </table>
</div>

### 2. VLA Baseline Success (GR00T-N1)
We successfully trained GR00T-N1 to imitate both styles. Despite early protocol mismatches, the "Parity Refactor" stabilized the inference stack, allowing GR00T to execute both behaviors reliably.

<div align="center">
  <table>
    <tr>
      <th>VLA: Grasp Execution</th>
      <th>VLA: Cup Execution</th>
    </tr>
    <tr>
      <td><video src="https://github.com/vedpatwardhan/le-probe/raw/main/assets/vla_grasp.mp4?raw=true" controls width="100%"></video></td>
      <td><video src="https://github.com/vedpatwardhan/le-probe/raw/main/assets/vla_cup.mp4?raw=true" controls width="100%"></video></td>
    </tr>
  </table>
</div>

### 3. LeWM Challenges (The Discriminability Gap)
LeWM, despite training with a large softrank, failed to sufficiently discriminate the goal state from non-goal states in the latent space. 

<div align="center">
  <h3>LeWM: Grasp Execution</h3>
  <video src="https://github.com/vedpatwardhan/le-probe/raw/main/assets/lewm_grasp.mp4?raw=true" controls width="100%"></video>
</div>

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
