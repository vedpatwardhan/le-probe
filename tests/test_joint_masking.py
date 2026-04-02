import pytest
import numpy as np


def test_ik_mask_strictness(sim):
    """
    Ensures that solve_ik only modifies whitelisted joints in the MuJoCo state.
    """
    # 1. Reset to home first
    sim.reset_env()
    home_q = sim.model.qpos0.copy()

    # 2. Define an extreme reach target to force global IK movement
    pos = np.array([0.5, -0.2, 1.1])
    quat = np.array([0.707, 0, 0.707, 0])

    # 3. Solve IK (internally respects authorized joints)
    q_result = sim.solve_ik(pos, quat)

    # 4. Global Verification
    # Every changed joint MUST be in the v_allowed_mask (ignoring small integration noise)
    changed_indices = []
    for i in range(sim.model.nq):
        if abs(q_result[i] - home_q[i]) > 1e-3:
            changed_indices.append(i)

    # CRITICAL ASSERTION: All changed indices must be authorized
    authorized_qpos_indices = set()
    for prot_idx, j_id in enumerate(sim.protocol_joint_ids):
        if j_id != -1 and sim.v_allowed_mask[prot_idx] > 0.5:
            authorized_qpos_indices.add(sim.model.jnt_qposadr[j_id])

    unauthorized_changes = [
        i for i in changed_indices if i not in authorized_qpos_indices
    ]

    # We allow the root/free joint to change (indices 0-6 usually)
    # but for this specific test we focus on robot bones.
    # v_allowed_mask is specifically for robot joints.

    assert (
        len(unauthorized_changes) == 0
    ), f"Unauthorized joint movement detected! qpos indices: {unauthorized_changes}"
    assert len(changed_indices) > 0, "IK failed to move any joints at all."


def test_reset_mask_strictness(sim):
    """
    Ensures that reset_env only randomizes whitelisted joints.
    """
    # 1. Start from home
    sim.reset_env()
    pre_reset_q = sim.data.qpos.copy()

    # 2. Trigger reset
    sim.reset_env()
    q_result = sim.data.qpos.copy()

    # 3. Check resulting pose against the snapshot
    # Only joints with randomization defined in reset_env should move.
    leaks = []
    # protocol_joint_ids maps protocol index -> mujoco joint id
    # we need a reverse map or a way to check if qpos index i is authorized
    authorized_qpos_indices = set()
    for prot_idx, j_id in enumerate(sim.protocol_joint_ids):
        if j_id != -1 and sim.v_allowed_mask[prot_idx] > 0.5:
            authorized_qpos_indices.add(sim.model.jnt_qposadr[j_id])

    for i in range(sim.model.nq):
        # 1e-2 threshold for randomization
        if abs(q_result[i] - pre_reset_q[i]) > 1e-2:
            # If it moved and it's not authorized, it's a leak
            # (ignoring indices for objects like cube_joint)
            if i not in authorized_qpos_indices:
                # check if this is a robot joint first
                found_robot_j = False
                for j in range(sim.model.njnt):
                    if (
                        sim.model.jnt_qposadr[j] == i
                        and "cube"
                        not in sim.model.names[sim.model.name_jntadr[j] :].decode()
                    ):
                        found_robot_j = True
                        break
                if found_robot_j:
                    leaks.append(i)

    assert len(leaks) == 0, f"Reset randomization leaked! Indices: {leaks}"
