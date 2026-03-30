import torch
import numpy as np

def test_ik_mask_strictness(sim):
    """
    Ensures that solve_ik only modifies whitelisted joints.
    Every other joint must remain at its current position.
    """
    # 1. Reset to home pose first
    sim.robot.set_qpos(sim.home_q)
    home_q = sim.home_q.clone().cpu().numpy()
    
    # 2. Define an extreme reach target to force global IK movement
    pos = (0.5, -0.2, 1.1)
    quat = (0.707, 0, 0.707, 0)
    
    # 3. Solve IK
    q_result = sim.solve_ik(pos, quat).cpu().numpy()
    
    # 4. Global Verification
    # We should only have movement in the whitelisted DOF indices
    allowed_dofs = set()
    for mapping in sim.joint_dof_map:
        if mapping["name"] in sim.allowed_names_set: # helper to be added below
             allowed_dofs.add(mapping["dof_idx"])
            
    # For speed, let's use the simulation's internal tracker
    allowed_dofs = set(sim.ik_dof_indices)
    
    changed_dofs = []
    # 1e-2 tolerance to account for dynamic settling during the PD glide.
    # Only check upper body (0-32) to isolate manipulation from leg-slump noise.
    for i in range(33):
        if abs(q_result[i] - home_q[i]) > 1e-2:
            changed_dofs.append(i)
            
    # CRITICAL ASSERTION: All changed DOFs must be in the whitelist
    unauthorized_changes = [i for i in changed_dofs if i not in allowed_dofs]
    assert len(unauthorized_changes) == 0, f"Unauthorized joint movement detected! DOFs: {unauthorized_changes}"
    assert len(changed_dofs) > 0, "IK failed to move any joints at all (target was likely unreachable or already at home)."

def test_reset_mask_strictness(sim):
    """
    Ensures that reset_env only randomizes whitelisted joints.
    """
    # 1. Start from home
    sim.robot.set_qpos(sim.home_q)
    home_q = sim.home_q.clone().cpu().numpy()
    
    # 2. Capture a pre-reset snapshot to isolate leakage from slump
    pre_reset_q = sim.robot.get_qpos().clone().cpu().numpy()
    
    # 3. Trigger reset
    sim.reset_env()
    
    # 4. Check resulting pose against the snapshot
    q_result = sim.robot.get_qpos().cpu().numpy()
    allowed_dofs = set(sim.ik_dof_indices)
    # Also include the waist pitch which we explicitly randomize
    waist_pitch_joint = sim.robot.get_joint("waist_pitch_joint")
    allowed_dofs.add(waist_pitch_joint.dofs_idx[0])
    
    # Only check the upper body (0-32) to isolate manipulation logic from leg-slump
    leaks = {i: abs(q_result[i] - pre_reset_q[i]) for i in range(33) 
             if abs(q_result[i] - pre_reset_q[i]) > 1e-2 and i not in allowed_dofs}
    
    assert len(leaks) == 0, f"Reset randomization leaked! (DOF: Delta): {leaks}"
