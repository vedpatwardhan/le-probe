# LeWM: LeRobot World Model & Oracle MPC

This module implements the **LeWM** (LeRobot World Model) training and inference stack, using a JEPA-based architecture for latent imagination and Oracle MPC for planning.

## ⚠️ Current Challenges: The Discriminability Gap

Our research shows that while LeWM can learn to predict video frames accurately, it struggles with the **Latent Discriminability Gap**:

<div align="center">
  <h3>Planning Audit (LeWM MPC)</h3>
  <p>Note the flailing and lack of clear progress towards the goal despite correct frame prediction in imagination.</p>
  <video src="https://github.com/vedpatwardhan/le-probe/raw/main/assets/lewm_grasp.mp4?raw=true" controls width="100%"></video>
</div>

- **Latent Confusion**: The world model often fails to distinguish the final goal state from intermediate states in the latent manifold, leading to "stalled" planning.
- **Reward Head Intervention**: We use an auxiliary **Reward Predictor** to provide a clearer gradient for the MPC solver. This has shown improvement in the robot's movement intent, though smoothness still trails behind VLA baselines.
- **High-DoF Control**: Coordinating 32 joints remains a significant challenge for the CEM solver in complex, multi-phase sequences.

## 🛠 Core Components
- [`lewm_server.py`](lewm_server.py): Oracle MPC server hosting the JEPA model and CEM solver.
- [`goal_mapper.py`](goal_mapper.py): The "Brain"—manages latent goal memory and cost manifolds.
- [`train_lewm.py`](train_lewm.py): Unified training script for the world model and reward head.
- [`diagnose_mpc.py`](diagnose_mpc.py): Vectorized audit tool for measuring latent improvement over 150+ episodes.
