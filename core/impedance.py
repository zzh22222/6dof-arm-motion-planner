"""
Cartesian Impedance Controller (Operational Space Elasticity & Compliance)
for 6-DOF Robotic Manipulator.

Physics Principles:
Operational Space Virtual Spring-Damper Wrench:
    F_cart = K_p * (x_d - x) + D_p * (v_d - v)
    T_cart = K_o * e_orientation + D_o * (w_d - w)
    W_cart = [F_cart; T_cart] in R^6

Joint Torque Mapping via Analytic Jacobian Transpose:
    tau_task = J^T(q) * W_cart

Full Dynamic Torque Control:
    tau = tau_task + qfrc_bias(q, qdot) + tau_null
    where tau_null = (I - J^T * (J^#)^T) * [K_null * (q_rest - q) - D_null * qdot]
"""

import numpy as np
import mujoco

class CartesianImpedanceController:
    def __init__(self, model, data, site_name="peg_tip"):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        
        # Arm DOF indices (first 6 joints)
        self.arm_dof = 6
        
        # Preferred ergonomic rest posture (Home posture)
        self.q_rest = np.array([0.0, 0.45, 0.95, 0.0, 0.65, 0.0], dtype=np.float64)
        
        # Default Cartesian Translational Stiffness (N/m) & Damping (N*s/m)
        self.kp_trans = np.array([300.0, 300.0, 300.0], dtype=np.float64)
        self.kd_trans = np.array([30.0, 30.0, 30.0], dtype=np.float64)
        
        # Default Cartesian Rotational Stiffness (N*m/rad) & Damping (N*m*s/rad)
        self.kp_rot = np.array([25.0, 25.0, 15.0], dtype=np.float64)
        self.kd_rot = np.array([3.0, 3.0, 2.0], dtype=np.float64)
        
        # Null-space posture gains
        self.k_null = 10.0
        self.d_null = 2.0
        
        # Torque limits (Nm) for safety
        self.torque_limits = np.array([150.0, 200.0, 150.0, 80.0, 60.0, 40.0])

    def set_stiffness(self, trans_stiffness, rot_stiffness=None, trans_damping=None, rot_damping=None):
        """Configure Cartesian stiffness and damping matrices."""
        self.kp_trans = np.array(trans_stiffness, dtype=np.float64)
        if trans_damping is not None:
            self.kd_trans = np.array(trans_damping, dtype=np.float64)
        else:
            # Critical damping heuristic: D = 2 * sqrt(K)
            self.kd_trans = 2.0 * np.sqrt(np.maximum(1.0, self.kp_trans)) * 0.8
            
        if rot_stiffness is not None:
            self.kp_rot = np.array(rot_stiffness, dtype=np.float64)
        if rot_damping is not None:
            self.kd_rot = np.array(rot_damping, dtype=np.float64)
        else:
            self.kd_rot = 2.0 * np.sqrt(np.maximum(1.0, self.kp_rot)) * 0.6

    def set_compliant_assembly_mode(self):
        """
        Anisotropic Compliance for Peg-in-Hole Assembly:
        - Soft in XY plane (120 N/m) to compliantly slide and self-center on chamfer
        - Stiff in Z axis (350 N/m) to drive peg downward
        - Soft in rotation (15 Nm/rad) to un-wedge upon jamming
        """
        self.set_stiffness(
            trans_stiffness=[120.0, 120.0, 350.0],
            rot_stiffness=[15.0, 15.0, 10.0],
            trans_damping=[22.0, 22.0, 40.0],
            rot_damping=[2.5, 2.5, 1.8]
        )

    def set_rigid_baseline_mode(self):
        """
        High-stiffness rigid mode (causes wedging/jamming on misalignment).
        """
        self.set_stiffness(
            trans_stiffness=[2200.0, 2200.0, 2200.0],
            rot_stiffness=[200.0, 200.0, 150.0],
            trans_damping=[90.0, 90.0, 90.0],
            rot_damping=[15.0, 15.0, 10.0]
        )

    def set_interactive_spring_mode(self, k_trans=180.0):
        """
        Virtual 3D elastic spring mode for mouse pulling.
        """
        self.set_stiffness(
            trans_stiffness=[k_trans, k_trans, k_trans],
            rot_stiffness=[25.0, 25.0, 15.0],
            trans_damping=[20.0, 20.0, 20.0],
            rot_damping=[3.0, 3.0, 2.0]
        )

    def get_ee_pose(self):
        """Get current 3D position and 3x3 rotation matrix of tracking site."""
        pos = np.copy(self.data.site_xpos[self.site_id])
        rot = np.copy(self.data.site_xmat[self.site_id].reshape(3, 3))
        return pos, rot

    def get_jacobian(self):
        """Compute 6x6 geometric Jacobian for the arm joints."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        
        Jp = jacp[:, :self.arm_dof]
        Jr = jacr[:, :self.arm_dof]
        return np.vstack([Jp, Jr]) # (6, 6)

    def compute_orientation_error(self, R_curr, R_desired):
        """
        Computes 3D angular rotation error vector e_rot such that:
        tau_rot = K_rot * e_rot
        """
        # Cross product form of relative rotation matrix
        e_rot = 0.5 * (
            np.cross(R_curr[:, 0], R_desired[:, 0]) +
            np.cross(R_curr[:, 1], R_desired[:, 1]) +
            np.cross(R_curr[:, 2], R_desired[:, 2])
        )
        return e_rot

    def compute_torque(self, x_desired, R_desired, v_desired=None, w_desired=None):
        """
        Computes full Cartesian Impedance Control joint torque command:
        tau = J^T * W_cart + qfrc_bias + tau_null
        """
        x_curr, R_curr = self.get_ee_pose()
        
        # 1. Linear velocity & Angular velocity of site
        J = self.get_jacobian()
        qvel = self.data.qvel[:self.arm_dof]
        ee_vel_6d = J @ qvel
        v_curr = ee_vel_6d[:3]
        w_curr = ee_vel_6d[3:]
        
        if v_desired is None:
            v_desired = np.zeros(3)
        if w_desired is None:
            w_desired = np.zeros(3)
            
        # 2. Cartesian Spatial Errors
        e_pos = x_desired - x_curr
        e_vel = v_desired - v_curr
        e_rot = self.compute_orientation_error(R_curr, R_desired)
        e_ang_vel = w_desired - w_curr
        
        # 3. Virtual Spring-Damper Wrench (Task space)
        F_cart = self.kp_trans * e_pos + self.kd_trans * e_vel
        T_cart = self.kp_rot * e_rot + self.kd_rot * e_ang_vel
        W_cart = np.concatenate([F_cart, T_cart]) # (6,)
        
        # Clamp Cartesian force to realistic motor capability
        force_norm = np.linalg.norm(W_cart[:3])
        if force_norm > 120.0:
            W_cart[:3] *= (120.0 / force_norm)
            
        # 4. Map Wrench to Joint Space via Jacobian Transpose: tau_task = J^T * W
        tau_task = J.T @ W_cart
        
        # 5. Null-Space Posture Ergonomic Bias: (I - J^T * (J^#)^T) * tau_posture
        # Damped Pseudoinverse of J: J_pinv = (J^T J + lambda^2 I)^(-1) J^T
        lambda_sq = 1e-3
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + lambda_sq * np.eye(6))
        N = np.eye(self.arm_dof) - J.T @ J_pinv.T
        
        qpos = self.data.qpos[:self.arm_dof]
        tau_posture = self.k_null * (self.q_rest - qpos) - self.d_null * qvel
        tau_null = N @ tau_posture
        
        # 6. Analytical Bias Torque (Gravity + Coriolis Cancellation)
        tau_bias = np.copy(self.data.qfrc_bias[:self.arm_dof])
        
        # Total commanded motor torque
        tau = tau_task + tau_bias + tau_null
        
        # Clamp to physical motor limits
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        
        # Diagnostic tracking metrics
        info = {
            "pos_error_norm": np.linalg.norm(e_pos),
            "cartesian_force": np.linalg.norm(F_cart),
            "contact_force": np.linalg.norm(self.data.qfrc_constraint[:self.arm_dof]),
            "x_curr": np.copy(x_curr),
            "x_desired": np.copy(x_desired)
        }
        
        return tau, info
