# Interpretability: Probing the Latent Mystery

This module is the research frontier of **Le-Probe**. It focuses on answering the fundamental question: **"Why does the World Model fail where the VLA succeeds?"**

## 🔭 Research Agenda

My current findings show that LeWM struggles with **Latent Discriminability**—the inability to find a clear path to the goal in the latent manifold. I am building tools to probe this behavior:

1.  **Latent Trajectory Analysis**: Visualizing the "imagined" future states of the World Model to identify where the planner diverges from reality.
2.  **Circuit Tracing**: Mechanistic analysis of the transformer layers to understand how reach-to-grasp intent is encoded (or lost).
3.  **Reward Manifold Auditing**: Mapping the topology of the reward head predictions to find local minima and "dead zones" that trap the MPC solver.

### 🍎 Phase I: The SAE Harvest
To "crack" the 192d latent bottleneck, we perform a **Data Harvest**. This involves:
- **Instrumentation**: Attaching hooks to the `Predictor` and `Encoder` to capture internal activations.
- **Data Crop**: Running ~1,000 diverse simulation episodes (successes, near-misses, and failures).
- **Decomposition**: Training a **Sparse Autoencoder (SAE)** on this activation dataset to map dense latents to 12,000+ monosemantic physical features (e.g., "gripper-cube contact").

## 🔬 Goal
By probing the internals of the JEPA architecture, I aim to bridge the gap between "good imagination" (visual accuracy) and "good action" (motor control), enabling World Models to handle the 32-DoF complexity of the GR-1 platform.

<div align="center">
  <img src="../assets/interpretability_architecture.png" width="70%" style="border-radius: 12px; margin-top: 20px;">
  <p><i>LeWM Interpretability: Mechanistic Analysis & Causal Intervention Stack</i></p>
</div>

---
*Status: Initial toolkit development in progress.*
