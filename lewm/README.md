# 🌍 World Model (LeWM) & Planning

> Probing behavior through latent imagination and Model Predictive Control.

This module contains the implementation of the **LeRobot World Model (LeWM)**, including training scripts, reward head tuning, and the MPC-based planning server.

## 🛠 Key Components

- **`lewm_server.py`**: The "Oracle" planning server that uses CEM (Cross-Entropy Method) to search for optimal action sequences in the latent space.
- **`train_lewm.py`**: Official training script for the world model with SIGReg and RA-LeWM (Reward-Aware) support.
- **`goal_mapper.py` & `goal_utils.py`**: Utilities for mapping visual observations to goal latents.
- **`lewm_data_plugin.py`**: A specialized data loader for training World Models on LeRobot datasets.

## 🔬 Research & Diagnostics

- **`tune_reward_head.py`**: Fine-tunes the reward predictor on small success/failure samples.
- **`diagnose_mpc.py`**: Visualizes the CEM search process and imagination rollouts.
- **`harvest_goals.py`**: Extracts goal snapshots from successful episodes.

---
*Part of the [Le-Probe](..) project.*
