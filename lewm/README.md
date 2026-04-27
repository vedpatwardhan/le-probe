# LeWM: LeRobot World Model & Oracle MPC

This module implements the **LeWM** (LeRobot World Model) training and inference stack, using a JEPA-based architecture for latent imagination and Oracle MPC for planning.

## ⚠️ Current Challenges: The Discriminability Gap

My research shows that while LeWM can learn to predict video frames accurately, it struggles with the **Latent Discriminability Gap**:

<div align="center">
  <h3>Planning Audit (LeWM MPC)</h3>
  <p>Note the flailing and lack of clear progress towards the goal despite correct frame prediction in imagination.</p>
  <img src="../assets/lewm_grasp.gif" width="320">
</div>

- **Latent Confusion**: The world model often fails to distinguish the final goal state from intermediate states in the latent manifold, leading to "stalled" planning.
- **Reward Head Intervention**: I use an auxiliary **Reward Predictor** to provide a clearer gradient for the MPC solver. This has shown improvement in the robot's movement intent, though smoothness still trails behind VLA baselines.

## 🚀 Workflows

### 1. Training
The model is trained using [**`LeWM_Training.ipynb`**](LeWM_Training.ipynb).
- **JEPA Training**: Performed under the `GR-1 Pickup Grasp` section.
- **Reward Head Training**: Performed under the `GR-1 Reward Pred` section.

After training, use [**`harvest_goals.py`**](harvest_goals.py) to harvest latent goal embeddings into a gallery.

**Pre-trained Artifacts:**
- [`gr1_reward_tuned_v2.ckpt`](https://drive.google.com/file/d/12YDes7GSQRWzQ-IMHbpq_64oWEoYj96V/view?usp=sharing): Reward-tuned checkpoint.
- [`goal_gallery.pth`](https://drive.google.com/file/d/1l-jdRkcwUUYxLcDiyDS6pb59M-CeZfSf/view?usp=sharing): Harvested latent goal gallery.

### 2. Inference
To test the World Model and MPC planner:

1. **LeWM MPC Server**: Start the server with the tuned model and gallery.
   ```bash
   python lewm/lewm_server.py --model gr1_reward_tuned_v2.ckpt --gallery goal_gallery.pth
   ```
2. **Simulation Host**: Start the MuJoCo environment.
   ```bash
   python lewm/simulation_lewm.py --host <host> --port <port>
   ```
3. **Execution**: Use [**`LEWM_E2E.ipynb`**](LEWM_E2E.ipynb) to manage the planning loop and visualize latent predictions.

## 🛠 Core Components
- [`lewm_server.py`](lewm_server.py): Standalone ZMQ server hosting the JEPA model and CEM solver.
- [`simulation_lewm.py`](simulation_lewm.py): MuJoCo simulation host for World Model testing.
- [`goal_mapper.py`](goal_mapper.py): Manages latent goal memory and manifold traversal.
- [`train_lewm.py`](train_lewm.py): Core training logic for the world model and reward head.
- [`harvest_goals.py`](harvest_goals.py): Utility to pre-compute goal embeddings.
