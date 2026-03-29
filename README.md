# GR-1 GR00T-N1.5 Inference & Teleoperation System

This repository contains the logic for running real-time inference and teleoperation for the GR-1 humanoid robot using the GR00T-N1.5 model.

## 🏗️ System Architecture
The system is divided into two parts:
1.  **Inference Server (Cloud/Colab)**: Runs the 3B-parameter GR00T-N1.5 model. Listens for observation payloads (images + state) over ZMQ.
2.  **Simulation Client (Local Mac)**: Runs the Genesis physics simulation, captures camera frames, and sends them to the server.

---

## 🚀 Quick Start (Inference)

### 1. Start the Inference Server (Colab)
Run the server pointing to your desired model.
```bash
# For the fine-tuned pickup model:
python -m gr1_gr00t.server_gr1 --model vedpatwardhan/GR00T-N1.5-finetuned-pickup

# For the base Nvidia model:
python -m gr1_gr00t.server_gr1 --model nvidia/GR00T-N1.5-3B
```

### 2. Setup the Rerun Tunnel (Local Mac)
To see the robot's vision in real-time, you must tunnel your local Rerun instance (Port 9876) to the cloud.
```bash
ssh -p 443 -R0:localhost:9876 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 tcp@pinggy.io
```
**Important**: Note the URL and port Pinggy gives you (e.g., `tcp://xxxx.pinggy.link:12345`).

### 3. Start the Simulation Client (Local Mac)
Update the Pinggy URL in `gr1_gr00t/simulation_gr1.py` and run:
```bash
python gr1_gr00t/simulation_gr1.py
```

---

## 🕹️ Teleoperation & Data Collection
If you need to record new demonstration data:
1.  Run **`teleop_ui.py`** to start the slider-based control interface.
2.  Run **`simulation.py`** to start the recording-enabled simulation environment.
3.  Toggle "Record" in the TUI to save episodes in `lerobot` format.

---

## 📂 Key Files
- **`server_gr1.py`**: The ZMQ server handling model loading and batch inference.
- **`simulation_gr1.py`**: The multi-step inference client with Genesis physics.
- **`simulation.py`**: The teleoperation simulator with `LeRobotManager` integration.
- **`gr1_config.py`**: The "Source of Truth" for joint limits and Rosetta protocol mappings.
- **`lerobot_manager.py`**: Utility for saving dataset episodes (images + states + actions).

---

## ⚠️ Common Troubleshooting
### "Transport Error" in Rerun
This happens if your Pinggy tunnel is pointing to the wrong local port. 
- **Correct**: `-R0:localhost:9876` (Rerun Port)
- **Incorrect**: `-R0:localhost:8000` (Default Pinggy Port)

### "Input Shape Mismatch"
Ensure the server and client agree on the number of cameras.
- **Legacy**: 4 cameras.
- **Pickup Task**: 5 cameras (including `world_wrist`).
