# Interpretability: Probing the Latent Mystery

This module is the research frontier of **Le-Probe**. It focuses on answering the fundamental question: **"Why does the World Model fail where the VLA succeeds?"**

## 🔭 Research Agenda

Our current findings show that LeWM struggles with **Latent Discriminability**—the inability to find a clear path to the goal in the latent manifold. We are building tools to probe this behavior:

1.  **Latent Trajectory Analysis**: Visualizing the "imagined" future states of the World Model to identify where the planner diverges from reality.
2.  **Circuit Tracing**: Mechanistic analysis of the transformer layers to understand how reach-to-grasp intent is encoded (or lost).
3.  **Reward Manifold Auditing**: Mapping the topology of the reward head predictions to find local minima and "dead zones" that trap the MPC solver.

## 🔬 Goal
By probing the internals of the JEPA architecture, we aim to bridge the gap between "good imagination" (visual accuracy) and "good action" (motor control), enabling World Models to handle the 32-DoF complexity of the GR-1 platform.

---
*Status: Initial toolkit development in progress.*
