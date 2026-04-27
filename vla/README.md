# VLA: Vision-Language-Action Baselines

This module hosts the **GR00T-N1** inference server and evaluation notebooks. My VLA implementation serves as the high-performance baseline for comparing World Model (LeWM) behaviors.

## 🏆 Current Performance

GR00T-N1 has been successfully stabilized to perform two distinct manipulation styles on a 32-DoF humanoid platform:

<div align="center">
  <h3>1. Grasp Pattern</h3>
  <p>Precision approach and pinch-grasp of the cube.</p>
  <img src="../assets/vla_grasp.gif" width="320">

  <h3>2. Cup Pattern</h3>
  <p>A "surrounding" movement optimized for containment rather than friction-based grasping.</p>
  <img src="../assets/vla_cup.gif" width="320">
</div>

### Implementation Success:
My 15k-step baseline was stabilized by achieving bit-perfect normalization parity with the training stack. This resolved early divergence issues where the model's reaching intent was correct but the physical execution was biased by unscaled joint values.

## 🚀 Workflows

### 1. Training
The model was trained using the behavioral cloning pipeline in [**`GR00T_N1_BC.ipynb`**](GR00T_N1_BC.ipynb).

Pre-trained model weights are available on Google Drive:
| Behavior | Pretrained Model Link |
| --- | --- |
| **Cup** | [Download](https://drive.google.com/drive/folders/1f5p6-5p6_20PpfbONcq-n5T1P7DhHfBw?usp=sharing) |
| **Grasp** | [Download](https://drive.google.com/drive/folders/1077_msVzs_8AQPaEbDm6XPiq8T_hxirp?usp=sharing) |

### 2. Inference
To run the stabilized VLA policy in simulation:

1. **Inference Server**: Start the server (can be bridged via Pinggy).
   ```bash
   python vla/gr00t_server.py --weights <path_to_weights>
   ```
2. **Simulation Host**: Start the MuJoCo environment.
   ```bash
   python vla/simulation_vla.py --host <host> --port <port> --chunks <num_chunks>
   ```
3. **Execution**: Use [**`GR00T_N1_E2E.ipynb`**](GR00T_N1_E2E.ipynb) to trigger rollouts and monitor performance.

## 📁 Key Files
- [`gr00t_server.py`](gr00t_server.py): The ZMQ inference host.
- [`simulation_vla.py`](simulation_vla.py): MuJoCo simulation environment for VLA testing.
- [`GR00T_N1_E2E.ipynb`](GR00T_N1_E2E.ipynb): End-to-end evaluation pipeline.
- [`GR00T_N1_BC.ipynb`](GR00T_N1_BC.ipynb): Behavioral Cloning training and audit logs.
