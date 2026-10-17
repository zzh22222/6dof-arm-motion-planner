"""
Kinematics Solver for 6-DOF Manipulator.
Provides:
1. Forward Kinematics (FK)
2. Analytical Jacobian extraction (Translational & Rotational)
3. Robust Damped Least-Squares (DLS) Inverse Kinematics (IK) with Null-Space Posture Bias
"""

import numpy as np
import mujoco

class ArmKinematics:
    def __init__(self, model, data, arm_joint_names=None, ee_site_name="end_effector"):
        self.model = model
        self.data = data
        self.ee_site_name = ee_site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        
        if arm_joint_names is None:
            self.arm_joint_names = [f"joint{i+1}" for i in range(6)]
        else:
            self.arm_joint_names = arm_joint_names
            
        self.arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names]
        self.arm_qpos_indices = [model.jnt_qposadr[j_id] for j_id in self.arm_joint_ids]
        self.arm_dof_indices = [model.jnt_dofadr[j_id] for j_id in self.arm_joint_ids]
        
        # Joint limits
        self.joint_limits = np.array([model.jnt_range[j_id] for j_id in self.arm_joint_ids])
        
        # Preferred ergonomic rest posture (Home posture)
        self.q_rest = np.array([0.0, -0.6, 1.2, 0.0, 0.9, 0.0])

    def get_joint_positions(self):
        """Read current arm joint angles."""
        return np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])

    def get_ee_position(self):
        """Read current 3D position of end-effector site."""
        return np.copy(self.data.site_xpos[self.site_id])

    def set_joint_positions(self, q):
        """Set arm joint angles directly into physics state."""
        for idx, val in zip(self.arm_qpos_indices, q):
            self.data.qpos[idx] = val
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, q=None):
        """
        Compute end-effector position and orientation matrix for joint angles q.
        Returns: (pos, rot_mat)
        pos: [x, y, z]
        rot_mat: 3x3 rotation matrix
        """
        if q is not None:
            # Save current state
            saved_q = self.get_joint_positions()
            self.set_joint_positions(q)
            
        pos = np.copy(self.data.site_xpos[self.site_id])
        rot_mat = np.copy(self.data.site_xmat[self.site_id].reshape(3, 3))
        
        if q is not None:
            # Restore state
            self.set_joint_positions(saved_q)
            
        return pos, rot_mat

    def get_jacobian(self):
        """
        Compute 6x6 analytical Jacobian for the 6 arm joints.
        Top 3 rows: translational Jacobian (Jp)
        Bottom 3 rows: rotational Jacobian (Jr)
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        
        # Extract only arm DOFs
        Jp_arm = jacp[:, self.arm_dof_indices]
        Jr_arm = jacr[:, self.arm_dof_indices]
        
        J = np.vstack([Jp_arm, Jr_arm])
        return J

    def solve_ik(self, target_pos, target_rot=None, q_init=None, 
                 max_iter=100, tol_pos=2e-3, tol_rot=2e-2, damping=0.015):
        """
        Damped Least-Squares (DLS) Inverse Kinematics with Null-Space Posture Bias.
        
        target_pos: [x, y, z]
        target_rot: 3x3 rotation matrix, or None if only position matters
        q_init: initial joint guess. If None, uses current robot posture.
        damping: Levenberg-Marquardt damping lambda (prevents singularity explosion)
        """
        saved_q = self.get_joint_positions()
        
        if q_init is None:
            q = np.copy(saved_q)
        else:
            q = np.copy(q_init)
            
        success = False
        k_null = 0.5  # Null space gain towards q_rest
        
        for iteration in range(max_iter):
            self.set_joint_positions(q)
            current_pos, current_rot = self.forward_kinematics()
            
            # Position error
            e_pos = target_pos - current_pos
            err_pos_norm = np.linalg.norm(e_pos)
            
            if target_rot is not None:
                # Orientation error via rotation matrix difference (skew-symmetric vector)
                rot_err_mat = current_rot.T @ target_rot
                # Extract rotation vector (axis-angle representation)
                e_rot_local = 0.5 * np.array([
                    rot_err_mat[2, 1] - rot_err_mat[1, 2],
                    rot_err_mat[0, 2] - rot_err_mat[2, 0],
                    rot_err_mat[1, 0] - rot_err_mat[0, 1]
                ])
                e_rot = current_rot @ e_rot_local
                err_rot_norm = np.linalg.norm(e_rot)
                
                # Stack 6D task error
                delta_x = np.concatenate([e_pos, e_rot])
                J = self.get_jacobian()
                
                if err_pos_norm < tol_pos and err_rot_norm < tol_rot:
                    success = True
                    break
            else:
                # 3D position-only task
                delta_x = e_pos
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
                J = jacp[:, self.arm_dof_indices]
                
                if err_pos_norm < tol_pos:
                    success = True
                    break

            # Damped Least-Squares Pseudoinverse: J_dls = J^T (J J^T + lambda^2 I)^(-1)
            JJT = J @ J.T
            damped_inv = np.linalg.inv(JJT + (damping ** 2) * np.eye(J.shape[0]))
            J_dls = J.T @ damped_inv
            
            # Primary task update
            delta_q_task = J_dls @ delta_x
            
            # Secondary task in Null Space: (I - J_pinv * J) * k_null * (q_rest - q)
            null_proj = np.eye(len(q)) - J_dls @ J
            delta_q_null = null_proj @ (k_null * (self.q_rest - q))
            
            # Total step with step-size limit to avoid overshooting
            step = delta_q_task + delta_q_null
            step_norm = np.linalg.norm(step)
            if step_norm > 0.2:
                step = step * (0.2 / step_norm)
                
            q += step
            
            # Clamp to physical joint limits
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # Restore original state in physics engine
        self.set_joint_positions(saved_q)
        
        return q, success
