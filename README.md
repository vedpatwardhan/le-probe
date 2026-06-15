# Le-Probe: Latent Topology Audits of LeWorldModel Representations for Humanoid Manipulation

[![Paper Status](https://img.shields.io/badge/Status-Under%20Review%20at%20CoRL%202026-blue)](https://drive.google.com/file/d/1-LUV945XR-FT33T3r6ZK4z1ag9cPRhd_/view?usp=drive_link)

This repository contains the code and diagnostic suite for **Le-Probe**, a diagnostic workflow that audits encoder latent topology (using PCA/UMAP on training rollouts and static task-hull probes) and relates geometry to planning failures in Joint-Embedding Predictive Architectures (specifically [LeWorldModel](https://arxiv.org/abs/2603.19312)) on a 32-DoF humanoid robot manipulation task.

Our diagnostics inspect why latent Model-Predictive Control (MPC) search succeeds or fails, tracing the effects of cumulative representation upgrades on both global geometry and local attribution circuits.

---

## 1. What Le-Probe Does

Le-Probe is a **diagnostic workflow** over encoder latents. It audits representation topology and relates it to planning failures using two primary paradigms:

1. **Training Manifold Audits:** Visualizing and analyzing the global topology of representations harvested from training trajectories (using UMAP, t-SNE, and PCA).
2. **Static Workspace Probes:** Evaluating 500 encode-only, out-of-distribution physical states generated inside the workspace polytope to test semantic partitioning (lateral location, distance to cube, and pose clusters).

<div align="center">
  <img src="assets/architecture_diagram.png" width="800" alt="Le-Probe Architecture Variants">
  <p><em>Unified architecture mapping the four evaluated representation variants.</em></p>
</div>

---

## 2. The Representation Ladder

We evaluate four checkpoints trained on the same tabletop red-cube pickup task (200 teleoperated GR-1 episodes in MuJoCo, 5 camera views, ~6.4k frames) under a shared latent planner (CEM, horizon $H=4$) to isolate the impact of representation design:

* **(a) Single-View RGB (Baseline):** The standard LeWorldModel mapping center-camera RGB to a 192-d embedding.
* **(b) Multi-View RGB:** Shared encoder across 5 camera views (center, left, right, top, wrist) with late linear fusion.
* **(c) Multi-View RGB + Skeletal Priors:** 4-channel input patch embeddings (RGB + skeletal lines) with perceptual shaping masking (10% skeleton-only, 5% counterpart-view masking).
* **(d) Multi-View + Skeletal + DINOv3 Waypoints:** The same backbone supplemented with a parallel, training-only frozen DINOv3 pathway and subgoal head to anchor phase goals.

---

## 3. Key Findings & Achievements

Our evaluation reveals a nuanced picture of representation learning in joint-embedding world models:

### Finding A: Trajectory Geometry vs. Control Quality
* **Global Alignment:** As stronger inductive biases are added, the training-time manifold transitions from disconnected, episode-isolated "worms" to a unified, directional "early-to-late highway" (with the strongest global phase organization visible in the DINOv3-supervised variant).
* **MPC Bottleneck:** Despite cleaner UMAP trajectory geometry, closed-loop latent MPC remains brittle during contact-rich phases (pinch and lift), indicating that improved global training topology does not automatically guarantee robust control.

<div align="center">
  <table>
    <tr>
      <th>Single-View RGB</th>
      <th>Multi-View RGB</th>
      <th>Skeletal Priors</th>
      <th>DINOv3 Waypoints</th>
    </tr>
    <tr>
      <td><img src="assets/lewm_grasp.gif" width="180" alt="Single-View RGB rollout"></td>
      <td><img src="assets/lewm_grasp_multiview.gif" width="180" alt="Multi-View RGB rollout"></td>
      <td><img src="assets/lewm_grasp_multiview_skeleton.gif" width="180" alt="Skeletal Priors rollout"></td>
      <td><img src="assets/lewm_grasp_multiview_skeleton_dino.gif" width="180" alt="DINOv3 Waypoints rollout"></td>
    </tr>
  </table>
</div>

### Finding B: Categorical Partitioning Fails Globally
When encoding 500 static poses inside the workspace hull, global embeddings fail to cluster cleanly by physical categories (e.g., table regions or distance bins). Silhouette scores in low-dimensional space remain near zero or negative across all models:

| Checkpoint Variant | Lateral Region Score | Distance Bin Score | Pose Cluster Score |
| :--- | :---: | :---: | :---: |
| **Single-View RGB** | -0.089 | -0.058 | -0.095 |
| **Multi-View RGB** | -0.058 | -0.084 | -0.036 |
| **Multi-View RGB + Skeletal Priors** | -0.033 | -0.068 | -0.039 |
| **Multi-View RGB + Skeletal + DINOv3** | -0.013 | -0.061 | -0.044 |

### Finding C: Local Sparse Features Retain Structure
While global clustering is poor, applying **Cross-Layer Transcoders (CLTs)** and **Integrated Gradients (IG)** to the static probes reveals checkpoint-dependent local circuits:
* Positive local feature Jaccard separation margins (0.12–0.34) exist across all checkpoints.
* Naive multi-view displays the sharpest lateral and distance circuit splits (only 3/15 and 5/15 node overlap between contrasting conditions), whereas DINOv3-waypoint models form more homogeneous local attribution circuits.

---

## 4. Repository Layout

* [`dataset/`](./dataset): MuJoCo teleoperation logging, dataset curation, skeletal/DINOv3 prior generation, and static probe generation.
* [`lewm/`](./lewm): Joint-embedding predictive architecture training, reward-head calibration, goal gallery harvesting, and the CEM latent planner server.
* [`interpretability/`](./interpretability): Manifold projection generation, static-probe separation metrics, CLT transcoder dictionary training, and Integrated Gradient path tracing.
* [`vla/`](./vla): GR00T-N1 baseline comparison suite.

---

## 5. Getting Started & Reproduction

### Installation
```bash
git clone --recursive https://github.com/vedpatwardhan/le-probe.git
cd le-probe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reference Notebooks
For step-by-step reproduction of the workflow:
* [`lewm/LeWM_Training.ipynb`](./lewm/LeWM_Training.ipynb): Handles joint-embedding training and checkpoint execution.
* [`lewm/LeWM_E2E.ipynb`](./lewm/LeWM_E2E.ipynb): Launches end-to-end simulation rollouts using the CEM latent planner.

Detailed CLI reproduction flags (including prior caches and server configurations) are documented in [`lewm/README.md`](./lewm/README.md).

---

## 6. Accessing Precomputed Models & Datasets

To support reuse and verification, we host the trained checkpoints and compiled harvests:

### Datasets & Galleries
* **Demonstration Dataset (`gr1_pickup_grasp`):** [Google Drive](https://drive.google.com/drive/folders/1yYMT7J_eRkQmXDq3tcisNd4kRSWeTI40)
* **Reward Predictions & Calibrations:** [Reward v1](https://drive.google.com/drive/folders/1QWra9dRJ9aceUqOpmj56OG8SaVUCVr-g) | [Reward v2](https://drive.google.com/drive/folders/1iwz_1LeEi4vbMWDeIXU_Pb6tVxDqcbNE)

### Checkpoints & Harvesters

| Model Variant | Checkpoint | Goal Gallery | Manifold Harvest |
| :--- | :--- | :--- | :--- |
| **Single-View RGB** | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1L0RE9V647-JduSCJ40y1TEI-N8MIO62D/view) | [goal_gallery.pth](https://drive.google.com/file/d/1CA9KxgnvHeJjslUOKoaxvmPV4TnhzWeS/view) | [manifold_data.pt](https://drive.google.com/file/d/18us_mOIVa2QgIP2VoISC-wpVzI7moCyV/view) |
| **Multi-View RGB** | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1VEEAa4vWcnqQN1PMK5422FK_1QJ0Hu74/view) | [goal_gallery.pth](https://drive.google.com/file/d/1ntMBODRRDP-bZDFUrbxli-3WxT4zveAv/view) | [manifold_data.pt](https://drive.google.com/file/d/1lqcmNQGiiECSPG4CM1h2c1S3JxwUQ_mP/view) |
| **MV + Skeletal** | [gr1_reward_tuned_v6.ckpt](https://drive.google.com/file/d/1W2UUco30AJE1ygjeGjRK1jFWB7PvGXEx/view) | [goal_gallery.pth](https://drive.google.com/file/d/1YEsGDwT1AvWetxS7vbLGL94xTOEDJtyP/view) | [manifold_data.pt](https://drive.google.com/file/d/19lxR0rJ-Oo7drudU_NyXQL3_cvlOGIcO/view) |
| **MV + Skel + DINOv3** | [gr1_reward_tuned_v1.ckpt](https://drive.google.com/file/d/1Yt1Q60yvvDPPFE3JjICq48ocOycUALGT/view) | [goal_gallery.pth](https://drive.google.com/file/d/1jpApbuPUHIAb3Ae87VzFAvFBVhVZr3X6/view) | [manifold_data.pt](https://drive.google.com/file/d/1Xhc9kMDilG3TpBA8GdDFLF4l7oe4j3Wz/view) |

Transcoder weights for dictionary evaluation across all variants are available in the [Transcoders Folder](https://drive.google.com/drive/folders/13Aw6iF1PfWqBR2CRh3A-wjqub6DP_Ty2).
