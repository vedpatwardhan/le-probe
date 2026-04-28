# Interpretability: Probing the Latent Mystery

This module is the research frontier of **Le-Probe**. It focuses on answering the fundamental question: **"Why does the World Model fail where the VLA succeeds?"**

## 🔭 Research Agenda

My current findings show that LeWM struggles with **Latent Discriminability**—the inability to find a clear path to the goal in the latent manifold. I am building tools to probe this behavior:

1.  **Latent Trajectory Analysis**: Visualizing the "imagined" future states of the World Model to identify where the planner diverges from reality.
2.  **Circuit Tracing**: Mechanistic analysis of the transformer layers to understand how reach-to-grasp intent is encoded (or lost).
3.  **Reward Manifold Auditing**: Mapping the topology of the reward head predictions to find local minima and "dead zones" that trap the MPC solver.

---

## 🏗 Toolkit Architecture

We have implemented a three-phase mechanistic interpretability stack that operates with **Zero-Impact Modularity** (using PyTorch hooks to avoid modifying the core `lewm` code).

### 🍎 Phase I: The SAE Harvest (`/sae`)
To "crack" the 192d latent bottleneck, we perform a **Data Harvest**.
- **Instrumentation**: `activation_harvester.py` attaches hooks to the `Predictor` and `Encoder`.
- **Data Crop**: `harvest_activations.py` runs ~1,000 diverse simulation episodes.
- **Decomposition**: `train_sae.py` trains a **Sparse Autoencoder** to map dense latents to 12,000+ monosemantic physical features (e.g., "gripper-cube contact").

### ⚡ Phase II: Circuit Tracing (`/clt`)
Once features are isolated, we use **Cross-Layer Transcoders (CLTs)** to understand the model's logic.
- **Mechanism**: `clt_model.py` maps features from one layer/time-step directly to the next.
- **Goal**: Tracing "verbs" (e.g., *how* an action causes a state change) by linearizing the Predictor's internal MLP.

### 🎮 Phase III: Causal Intervention (`/steering`)
The final phase applies this knowledge to active control.
- **Mechanism**: `latent_steering.py` allows for real-time feature ablation or amplification during the CEM unroll.
- **Use Case**: Surgically removing "Hallucination" features or amplifying "Stability" features to guide the robot's physical intent.

---

### 🛠 Why Custom Harvester vs. SAELens-V?
We use a custom implementation for the **Harvest** to ensure:
1.  **Zero-Impact Modularity**: No changes to the `le_wm` source code via external hooks.
2.  **Simulation Integration**: Direct coupling with the MuJoCo physics step.
3.  **JEPA Compatibility**: Handling Gaussian-regularized (SIGReg) latents which differ from standard LLM residual streams.

*Note: Once harvested, the data can be exported to SAELens-V for advanced feature dashboarding and circuit analysis.*

## 🔬 Results
After training the CLT, we successfully isolated three core mechanistic features that define the GR-1 pickup sequence in the `gr1_pickup_grasp` dataset:

| Feature | Label | Max Act. | Episode | Frame Index | Phase Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **358** | **Spatial Lockdown** | **2.0461** | 111 | 27 | Lift (Post-Grip) |
| **90** | **Tactile Engagement** | **1.5157** | 115 | 23 | Grasp (Coupling) |
| **743** | **Alignment Precision** | **1.5508** | 19 | 25 | Grasp-to-Lift Handover |

Following are the plots demonstrating the transition of states triggering the above features:

<div align="center">
  <p>Spatial Lockdown</p>
  <img src="../assets/spatial_lockdown.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
</div>

<div align="center">
  <p>Tactile Engagement</p>
  <img src="../assets/tactile_engagement.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
</div>

<div align="center">
  <p>Alignment Precision</p>
  <img src="../assets/alignment_precision.png" width="100%" style="border-radius: 12px; margin-bottom: 20px;">
</div>

## 🚀 Research Roadmap: Next Steps
The discovery of these interpretable features allows us to audit the effects of key architectural changes:

1.  **Multi-View Latents**: Transitioning from single-view (`world_center`) to the full 5-view stack used by GR00T. This is expected to harden Feature 358 (Spatial Lockdown) against visual occlusion.
2.  **Reachability Constraints**: Integrating kinematic polytopes (e.g., PyCapacity) to prevent the MPC solver from exploring non-physical "folding" maneuvers that the dataset cannot currently penalize.
3.  **Latent Steering**: Closing the causal loop by using Feature 90 (Tactile Engagement) as a reward booster during real-time inference.

---
*Status: Framework operational. Canonical features 90, 358, and 743 verified.*
