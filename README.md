# GR-1 GR00T-N1.5: Modular MuJoCo Foundation 🦾

This repository contains the high-fidelity physics environment and inference lifecycle for the GR-1 humanoid robot, now powered by **MuJoCo** and the **Minkowski** IK solver.

## 🏗️ System Architecture: The Modular Split
The system is decoupled into specialized drivers sharing a common physical foundation.

1.  **`simulation_base.py` (The Physical Core)**: The "Source of Truth" for robot physics. Handles MuJoCo XML loading, Minkowski Triple-Link IK solving, and state normalization.
2.  **`simulation_vla.py` (Autonomous Mission)**: A proactive **REQ Client** designed for headless/Colab execution. It drives the "Sense -> Plan -> Act" loop for a 10-chunk (160 step) mission.
3.  **`simulation_teleop.py` (Dashboard Server)**: A reactive **REP Server** specifically for interactive control and dataset recording via the Streamlit UI.
4.  **`gr00t_server.py` (The Brain)**: A passive inference brain that processes 5-camera observations and returns 16-action chunks.

---

## 🚀 Headless Autonomy (VLA Mission)
Designed for Colab or remote environments where no UI is needed.

### 1. Start the Inference Server
```bash
uv run gr1_gr00t/gr00t_server.py --model nvidia/GR00T-N1.5-3B
```

### 2. Launch the Mission Driver
```bash
uv run gr1_gr00t/simulation_vla.py --instruction "Pick up the red cube" --chunks 10
```

---

## 🕹️ Interactive Teleop & Data Collection
For manual joint control, IK calibration, and `LeRobot` dataset generation.

### 1. Start the Teleop Sim Server (Port 5556)
```bash
uv run gr1_gr00t/simulation_teleop.py
```

### 2. Start the Streamlit Dashboard
```bash
streamlit run gr1_gr00t/teleop_ui.py
```

### 3. Record Data
- Click **"Start Recording"** in the Dashboard.
- Use sliders or the **🎯 IK Pickup** button to perform tasks.
- Click **"Stop Recording"** to finalize the episode and push it to the Hub.

---

## 📂 Key Files
- **`gr1_config.py`**: Central registry for `XML_PATH`, `SCENE_PATH`, and joint normalization limits.
- **`sim_assets/`**: High-fidelity MuJoCo XMLs for the robot and pickup environments.
- **`lerobot_manager.py`**: Background uploader for episodic dataset management.
- **`teleop_joints.txt` / `ik_joints.txt`**: Whitelists for joint authorization during control.

---

## ⚠️ Requirements
- **OS**: macOS (Local) / Linux (Colab)
- **Physics**: `mujoco`, `mink`
- **Networking**: `zmq`, `msgpack`
- **Visualization**: `rerun-sdk` (Ensure Port 9876 is open/tunneled)
