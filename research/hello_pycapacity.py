import numpy as np
import sys

try:
    import pycapacity.robot as robot

    PYCAPACITY_AVAILABLE = True
except ImportError:
    PYCAPACITY_AVAILABLE = False
    print(
        "[!] pycapacity not found. Please run: .venv/bin/python -m pip install pycapacity"
    )
    sys.exit(1)


def hello_pycapacity():
    print("--- pycapacity Hello World ---")

    # 1. Define a simple Jacobian for a 2-link planar arm
    # Ensure it's a FLOAT64 numpy array
    # J = [[ -L1*sin(q1)-L2*sin(q1+q2), -L2*sin(q1+q2) ],
    #      [  L1*cos(q1)+L2*cos(q1+q2),  L2*cos(q1+q2) ]]
    # At q1=0, q2=pi/4 (arm bent):
    J = np.array([[-0.707, -0.707], [1.707, 0.707]], dtype=float)

    # 2. Define Joint Velocity Limits (dq in rad/s)
    # Ensure these are also numpy arrays
    dq_max = np.array([1.0, 1.0], dtype=float)
    dq_min = np.array([-1.0, -1.0], dtype=float)

    print(f"Jacobian Matrix J:\n{J}")
    print(f"Joint Velocity Limits: {dq_min} to {dq_max}")

    # 3. Calculate the Velocity Polytope
    print("\nCalculating Velocity Polytope...")
    try:
        poly = robot.velocity_polytope(J, dq_min, dq_max)

        # 4. Display Results
        from scipy.spatial import ConvexHull

        # For a 2D arm, ConvexHull(vertices).volume returns the area
        # For a 3D arm, it returns the volume
        hull = ConvexHull(poly.vertices.T)
        print(f"✅ Polytope Volume/Area: {hull.volume:.4f}")
        print(f"📍 Polytope Vertices:\n{poly.vertices}")
    except Exception as e:
        print(f"❌ Polytope calculation failed: {e}")
        import traceback

        traceback.print_exc()

    # Conceptual explanation
    print("\n[Concept]")
    print("The 'Volume' in 2D is the area of reachability.")
    print(
        "With J = [[0, 0], [2, 1]], the arm is technically singular in X (first row is 0)."
    )
    print("Notice how the vertices only vary in the second dimension (Y).")
    print("This perfectly illustrates how pycapacity detects lost degrees of freedom!")


if __name__ == "__main__":
    hello_pycapacity()
