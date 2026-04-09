import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import pycapacity.robot as robot
import os
import sys

# Ensure projects paths are included
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gr1_config import SCENE_PATH, COMPACT_WIRE_JOINTS


def gr1_reachability_visualizer():
    print("--- GR-1 Reachability Visualizer (MuJoCo Mode) ---")

    # 1. Load Model
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Put robot in a non-singular pose (e.g. lift right arm)
    # Right shoulder roll joint is index 17 in some protocols,
    # but we'll manually set a bend to elbows.
    for i in range(model.nu):
        data.ctrl[i] = 0.1  # Slight bias

    # Step simulation to update kinematics
    mujoco.mj_step(model, data, 10)

    # 2. Identify Target Site (Right Hand)
    # We'll use the body 'right_hand_pitch_link' center
    ee_name = "right_hand_pitch_link"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)

    # 3. Extract Jacobian (3xNV)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac, None, body_id)

    # 4. Joint Velocity Limits
    # We use 1.0 rad/s for normalization across all DOFs
    dq_max = np.ones(model.nv)
    dq_min = -np.ones(model.nv)

    # 5. Calculate Polytope
    print("Calculating 3D Velocity Polytope...")
    try:
        # We calculate the velocity polytope for [dx, dy, dz]
        poly = robot.velocity_polytope(jac, dq_min, dq_max)

        # Calculate volume
        hull = ConvexHull(poly.vertices.T)
        print(f"✅ Reachability Volume: {hull.volume:.6f}")

    except Exception as e:
        print(f"❌ Polytope calculation failed: {e}")
        return

    # 6. Visualization (Matplotlib 3D)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot Vertices
    verts = poly.vertices.T
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color="r", alpha=0.5)

    # Plot the Convex Hull Faces
    for simplex in hull.simplices:
        poly_face = Poly3DCollection([verts[simplex]], alpha=0.3)
        poly_face.set_facecolor("blue")
        poly_face.set_edgecolor("black")
        ax.add_collection3d(poly_face)

    ax.set_xlabel("dX (m/s)")
    ax.set_ylabel("dY (m/s)")
    ax.set_zlabel("dZ (m/s)")
    ax.set_title(f"GR-1 Hand Velocity Polytope\nVolume: {hull.volume:.6f}")

    # Set plot limits centered on 0
    lim = 1.5
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    output_path = "research/gr1_reachability.png"
    plt.savefig(output_path)
    print(f"📈 Plot saved to {output_path}")
    print("\n[Analysis]")
    print("The shape reflects the 'Anisotropy' of the robot's reach.")
    print("Elongated axes show directions where the robot has maximum speed/control.")


if __name__ == "__main__":
    gr1_reachability_visualizer()
