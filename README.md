# 🦾 cortex-gr1: Modular MuJoCo Foundation

This repository contains the high-fidelity physics environment and inference lifecycle for the GR-1 humanoid robot, now powered by **MuJoCo** and the **Minkowski** IK solver.

## 🏗️ System Architecture: The Modular Split
The system is decoupled into specialized drivers sharing a common physical foundation.

1.  **`simulation_base.py` (The Physical Core)**: The "Source of Truth" for robot physics.
2.  **`le_wm/` (The Brain)**: A git submodule containing the JEPA (Joint-Embedding Predictive Architecture) world model.
3.  **`train_lewm.py` (Proprioceptive Learning)**: Fine-tunes the world model on your local 64-D Rosetta datasets.
4.  **`simulation_vla.py` (Autonomous Mission)**: A proactive **REQ Client** designed for headless/Colab execution.
5.  **`gr00t_server.py` (VLA Inference)**: A passive inference brain that processes observations using pre-trained VLAs.

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
- **`le_wm/`**: Submodule containing the JEPA architecture core.
- **`train_lewm.py`**: The main entry point for proprioceptive fine-tuning on Rosetta-64 datasets.
- **`sim_assets/`**: High-fidelity MuJoCo XMLs for the robot and pickup environments.

---

## 🧠 World Model Training (JEPA)
To teach the robot's "Brain" how its body moves and affects the world, you can fine-tune the JEPA world model on your collected `LeRobot` datasets.

### 1. Initialize Submodule
```bash
git submodule update --init --recursive
```

### 2. Start Fine-tuning
Trains the 64-D action predictor while leveraging pre-trained vision weights.
```bash
uv run train_lewm.py
```

---

## ⚠️ Requirements
- **OS**: macOS (Local) / Linux (Colab)
- **Physics**: `mujoco`, `mink`
- **Dependencies**: `lerobot`, `einops`, `transformers`, `huggingface-hub`
- **Visualization**: `rerun-sdk` (Ensure Port 9876 is open/tunneled)
