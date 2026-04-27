# 📦 Dataset Management & Teleoperation

> High-fidelity data collection and preprocessing for the GR-1 robot.

This module provides the core tools for interacting with the MuJoCo simulation via human teleoperation, recording episodes into the LeRobot format, and synchronizing with the Hugging Face Hub.

## 🛠 Key Components

- **`teleop_ui.py`**: A premium Streamlit dashboard for real-time joint control and phase management.
- **`simulation_teleop.py`**: The ZMQ server driving the MuJoCo simulation and handling IK requests.
- **`lerobot_manager.py`**: Advanced wrapper for `LeRobotDataset` with support for "Smart Rewards" and cloud sync.
- **`simulation_replay.py`**: Validates dataset integrity by replaying recorded episodes.
- **`upload_dataset.py`**: Streamlined utility for pushing local datasets to the Hugging Face Hub.

## 🚀 Usage

### Real-time Teleoperation
1. Start the simulation server:
   ```bash
   .venv/bin/python dataset/simulation_teleop.py
   ```
2. Launch the dashboard:
   ```bash
   streamlit run dataset/teleop_ui.py
   ```

### Dataset Validation
To verify a recorded dataset:
```bash
.venv/bin/python dataset/simulation_replay.py --repo_id local/gr1_pickup
```

---
*Part of the [Le-Probe](..) project.*
