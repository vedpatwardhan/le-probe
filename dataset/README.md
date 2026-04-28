# Dataset Management & Teleoperation

This module handles the lifecycle of robotic data: from real-time MuJoCo teleoperation to LeRobot-formatted cloud datasets.

## 📊 Dataset Standards

After experimenting with 12, 32 and 52 frame episodes, I have standardized on **32-frame episodes** (recorded at 10Hz) to capture the full reach-to-grasp trajectory for our task. I maintain two primary behavioral variants:

<div align="center">
  <table>
    <tr>
      <th>Dataset: Grasp Pattern</th>
      <th>Dataset: Cup Pattern</th>
    </tr>
    <tr>
      <td><img src="assets/dataset_grasp.gif" width="320"></td>
      <td><img src="assets/dataset_cup.gif" width="320"></td>
    </tr>
  </table>
</div>

## 📐 Methodology

### 🖥 Teleoperation Interface

I use a custom-built Streamlit dashboard for real-time control, IK requests, and dataset auditing using the `teleop_ui.py` which is eventually visualized via Rerun.

<div align="center">
  <img src="../assets/teleop_dashboard.png" width="100%" style="border-radius: 8px;">
</div>

### 🛠 Key Components

- [**`teleop_ui.py`**](teleop_ui.py):
  - Streamlit dashboard for 32-DoF joint control and IK-assisted manipulation.
  - Contains options for controlling the robot manually through the sliders, using the IK solver for consistent movement, the recording of episodes into the dataset, monitoring of distance between the right hand fingers and the cube, etc.

- [**`simulation_teleop.py`**](simulation_teleop.py):
  - ZMQ server driving the MuJoCo simulation and handling 32-DoF IK requests and supports all the features listed in the `teleop_ui.py` dashboard.
  - Currently the IK solver operates in a grasp move, but essentially a few tweaks to the target coordinates can allow operations in a cup movement as well.

- [**`lerobot_manager.py`**](lerobot_manager.py):
  - Core recording logic in the LeRobot format.
  - Implements the 32-dim identity protocol and "Smart Reward" injection.
  - Rewards are currently assigned as an inverse of the distance between the right hand fingers and the cube (capped at `10`) using the `lerobot_manager.py`.

- [**`simulation_replay.py`**](simulation_replay.py):
  - Visual audit tool for replaying recorded episodes for verification.

- [**`upload_dataset.py`**](upload_dataset.py):
  - A script to upload the dataset to the Hugging Face Hub.

## 📊 Current Datasets

The following datasets have been curated and uploaded to the Hugging Face Hub:

- [**`gr1_pickup_grasp`**](https://huggingface.co/datasets/vedpatwardhan/gr1_pickup_grasp): Precision "pinch" grasp trajectories.
- [**`gr1_pickup_cup`**](https://huggingface.co/datasets/vedpatwardhan/gr1_pickup_cup): Robust "surrounding" containment trajectories.
- [**`gr1_reward_pred`**](https://huggingface.co/datasets/vedpatwardhan/gr1_reward_pred): Multi-behavioral data used to train the Reward Head. Wasn't curated using the IK Solver, but instead using Wild Randomization with the Snapshot button on the teleoperator for having a significant proportion of failing states.

## 🚀 Workflows

### 1. Data Collection
```bash
# Start the Rerun server
rerun

# Start Sim Server
.venv/bin/python dataset/simulation_teleop.py

# Start Dashboard
streamlit run dataset/teleop_ui.py
```

### 2. Dataset Upload
```bash
.venv/bin/python dataset/upload_dataset.py --repo_id <>
```
