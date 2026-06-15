# Le-Probe: Latent Topology Audits of LeWorldModel Representations

📄 **CoRL 2026 Submission Preprint:** [Read the Paper (Google Drive)](https://drive.google.com/file/d/1-LUV945XR-FT33T3r6ZK4z1ag9cPRhd_/view?usp=drive_link)
*(Currently under review)*

**Le-Probe** is a diagnostic workflow that audits encoder latent topology to analyze why latent Model-Predictive Control (MPC) succeeds or fails on a 32-DoF humanoid cube-pickup task. We evaluate how visual, kinematic, and subgoal inductive biases reshape Joint-Embedding Predictive Architectures (specifically [LeWorldModel](https://arxiv.org/abs/2603.19312)).

---

## 1. System Architecture & Representation Ladder

We evaluate four model variants under a shared latent planner (CEM, horizon $H=4$) to isolate the impact of representation design:

* **(a) Single-View RGB (Baseline):** Standard LeWorldModel mapping center-camera RGB to a 192-d embedding.
* **(b) Multi-View RGB:** Shared encoder across 5 camera views with late linear fusion.
* **(c) Multi-View RGB + Skeletal Priors:** 4-channel inputs (RGB + skeleton overlays) with perceptual shaping masking.
* **(d) Multi-View + Skeletal + DINOv3:** Added training-only frozen DINOv3 target path and subgoal head.

<div align="center">
  <img src="assets/architecture_diagram.png" width="750" alt="Le-Probe Architecture Variants">
</div>

---

## 2. The Le-Probe Auditing Workflow

Le-Probe audits representation topology post-training without modifying checkpoints or planners:
1. **Training Manifold Audits:** PCA, t-SNE, and UMAP mapping of training trajectory rollouts.
2. **Static Workspace Probes:** Semantic clustering evaluation of 500 out-of-distribution physical states.

---

## 3. Key Findings

### Finding A: Trajectory Manifold Progression
Adding stronger priors aligns trajectory rollouts, transitioning the latent space from disconnected episode "worms" to a unified, early-to-late phase highway.

<div align="center">
  <table>
    <tr>
      <th>Projection</th>
      <th>Single-View RGB</th>
      <th>Multi-View RGB</th>
      <th>Skeletal Priors</th>
      <th>DINOv3 Waypoints</th>
    </tr>
    <tr>
      <td><b>t-SNE</b></td>
      <td><img src="assets/manifold/manifold_3d_tsne.png" width="160"></td>
      <td><img src="assets/manifold/manifold_3d_multiview_tsne.png" width="160"></td>
      <td><img src="assets/manifold/manifold_3d_multiview_skeleton_tsne.png" width="160"></td>
      <td><img src="assets/manifold/manifold_3d_multiview_skeleton_dino_2_tsne.png" width="160"></td>
    </tr>
    <tr>
      <td><b>UMAP</b></td>
      <td><img src="assets/manifold/manifold_3d_umap.png" width="160"></td>
      <td><img src="assets/manifold/manifold_3d_multiview_umap.png" width="160"></td>
      <td><img src="assets/manifold/manifold_3d_multiview_skeleton_umap.png" width="160"></td>
      <td><img src="assets/manifold/manifold_3d_multiview_skeleton_dino_2_umap.png" width="160"></td>
    </tr>
  </table>
</div>

### Finding B: Latent MPC and Task Execution
Cleaner global trajectory manifolds do not automatically guarantee control success. MPC remains highly sensitive and prone to off-manifold drift during contact-rich pinch and lift phases.

<div align="center">
  <table>
    <tr>
      <th>Single-View RGB</th>
      <th>Multi-View RGB</th>
      <th>Skeletal Priors</th>
      <th>DINOv3 Waypoints</th>
    </tr>
    <tr>
      <td><img src="assets/lewm_grasp.gif" width="160"></td>
      <td><img src="assets/lewm_grasp_multiview.gif" width="160"></td>
      <td><img src="assets/lewm_grasp_multiview_skeleton.gif" width="160"></td>
      <td><img src="assets/lewm_grasp_multiview_skeleton_dino.gif" width="160"></td>
    </tr>
  </table>
</div>

### Finding C: Global Workspace Probes (Separation Fails)
Global latent spaces do not cluster cleanly by coarse semantic workspace categories (Silhouette scores remain near zero or negative).

| Checkpoint Variant | Lateral Region | Distance Bin | Pose Cluster |
| :--- | :---: | :---: | :---: |
| **Single-View RGB** | -0.089 | -0.058 | -0.095 |
| **Multi-View RGB** | -0.058 | -0.084 | -0.036 |
| **Multi-View RGB + Skeletal Priors** | -0.033 | -0.068 | -0.039 |
| **Multi-View RGB + Skeletal + DINOv3** | -0.013 | -0.061 | -0.044 |

### Finding D: Local Attribution Circuits (Positive Structure)
* **Local Features:** Cross-Layer Transcoders (CLTs) show positive separation margins (0.12–0.34) on the same probes, proving local features capture task structure.
* **Circuit Splitting:** Integrated Gradient (IG) pathways show distinct circuit splits for naive multi-view (3/15 node overlap) but highly homogeneous circuits under DINOv3 supervision.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Multi-View (Lateral Left)</b><br><img src="assets/circuits/lateral_table_region/multiview_left.png" width="220"></td>
      <td align="center"><b>Skeletal (Lateral Left)</b><br><img src="assets/circuits/lateral_table_region/skeleton_left.png" width="220"></td>
      <td align="center"><b>DINOv3 (Lateral Left)</b><br><img src="assets/circuits/lateral_table_region/dino_left.png" width="220"></td>
    </tr>
    <tr>
      <td align="center"><b>Multi-View (Approach)</b><br><img src="assets/circuits/distance_to_cube/multiview_approach.png" width="220"></td>
      <td align="center"><b>Multi-View (Near Table)</b><br><img src="assets/circuits/distance_to_cube/multiview_near_table.png" width="220"></td>
      <td align="center"><b>Skeletal (Pose Cluster 2)</b><br><img src="assets/circuits/pose_clusters/skeleton_pose_2.png" width="220"></td>
    </tr>
  </table>
</div>

---

## 4. Repository Layout

* [`dataset/`](./dataset): MuJoCo teleop logs, skeletal/DINOv3 prior extraction, and static probe generation.
* [`lewm/`](./lewm): JEPA encoder/predictor training, goal gallery harvests, and the CEM latent planner.
* [`interpretability/`](./interpretability): UMAP/t-SNE manifold projections, silhouette evaluations, CLT dictionary training, and IG circuits.
* [`vla/`](./vla): GR00T-N1 baseline evaluation suite.

---

## 5. Getting Started & Reproduction

```bash
git clone --recursive https://github.com/vedpatwardhan/le-probe.git
cd le-probe
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

* Launch training: [`lewm/LeWM_Training.ipynb`](./lewm/LeWM_Training.ipynb)
* Run evaluations: [`lewm/LeWM_E2E.ipynb`](./lewm/LeWM_E2E.ipynb)

See [`lewm/README.md`](./lewm/README.md) for CLI options.

---

## 6. Resources & Precomputed Models

### Datasets
* **Demonstrations (`gr1_pickup_grasp`):** [Google Drive](https://drive.google.com/drive/folders/1yYMT7J_eRkQmXDq3tcisNd4kRSWeTI40)
* **Reward Calibrations:** [Reward v1](https://drive.google.com/drive/folders/1QWra9dRJ9aceUqOpmj56OG8SaVUCVr-g) | [Reward v2](https://drive.google.com/drive/folders/1iwz_1LeEi4vbMWDeIXU_Pb6tVxDqcbNE)

### Models & Manifold Harvests

| Model Variant | Checkpoint | Goal Gallery | Manifold Harvest |
| :--- | :--- | :--- | :--- |
| **Single-View RGB** | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1L0RE9V647-JduSCJ40y1TEI-N8MIO62D/view) | [goal_gallery.pth](https://drive.google.com/file/d/1CA9KxgnvHeJjslUOKoaxvmPV4TnhzWeS/view) | [manifold_data.pt](https://drive.google.com/file/d/18us_mOIVa2QgIP2VoISC-wpVzI7moCyV/view) |
| **Multi-View RGB** | [gr1_reward_tuned_v2.ckpt](https://drive.google.com/file/d/1VEEAa4vWcnqQN1PMK5422FK_1QJ0Hu74/view) | [goal_gallery.pth](https://drive.google.com/file/d/1ntMBODRRDP-bZDFUrbxli-3WxT4zveAv/view) | [manifold_data.pt](https://drive.google.com/file/d/1lqcmNQGiiECSPG4CM1h2c1S3JxwUQ_mP/view) |
| **MV + Skeletal** | [gr1_reward_tuned_v6.ckpt](https://drive.google.com/file/d/1W2UUco30AJE1ygjeGjRK1jFWB7PvGXEx/view) | [goal_gallery.pth](https://drive.google.com/file/d/1YEsGDwT1AvWetxS7vbLGL94xTOEDJtyP/view) | [manifold_data.pt](https://drive.google.com/file/d/19lxR0rJ-Oo7drudU_NyXQL3_cvlOGIcO/view) |
| **MV + Skel + DINOv3** | [gr1_reward_tuned_v1.ckpt](https://drive.google.com/file/d/1Yt1Q60yvvDPPFE3JjICq48ocOycUALGT/view) | [goal_gallery.pth](https://drive.google.com/file/d/1jpApbuPUHIAb3Ae87VzFAvFBVhVZr3X6/view) | [manifold_data.pt](https://drive.google.com/file/d/1Xhc9kMDilG3TpBA8GdDFLF4l7oe4j3Wz/view) |

*Transcoder weights are available in the [Transcoders Folder](https://drive.google.com/drive/folders/13Aw6iF1PfWqBR2CRh3A-wjqub6DP_Ty2).*
