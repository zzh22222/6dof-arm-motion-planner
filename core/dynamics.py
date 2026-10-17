"""
Robotic Manipulator Dynamics & Zero-Gravity Lead-Through Teaching Engine.

Physics Principles:
Equation of Motion:
    M(q)*q_ddot + C(q, q_dot)*q_dot + g(q) = tau_motor + tau_ext

Zero-Gravity Hovering:
    Setting tau_motor = qfrc_bias(q, q_dot) - D_v * q_dot
    results in:
    M(q)*q_ddot + D_v * q_dot = tau_ext
    - Without human force (tau_ext = 0), q_ddot = -M^{-1} * D_v * q_dot (exponential stop, floating mid-air)
    - With human dragging (tau_ext != 0), the operator feels weightless inertia and viscous damping.

Trajectory Replay (Computed Torque Control):
    tau_motor = qfrc_bias + Kp * (q_ref(t) - q) + Kd * (q_dot_ref(t) - q_dot)
"""

import time
import numpy as np
import mujoco

class DynamicsController:
    """
    Computes analytical inverse dynamics, gravity cancellation, virtual joint damping,
    and soft joint-limit boundary cushions for 6-DOF manipulators.
    """
    def __init__(self, model, data, 
                 arm_dof=6, 
                 damping_diag=(1.2, 1.8, 1.2, 0.6, 0.4, 0.2),
                 kp_replay=(250.0, 300.0, 250.0, 120.0, 100.0, 60.0),
                 kd_replay=(25.0, 30.0, 25.0, 12.0, 10.0, 6.0)):
        self.model = model
        self.data = data
        self.arm_dof = arm_dof
        
        # Virtual Viscous Damping Matrix D_v (diagonal)
        self.damping = np.array(damping_diag, dtype=np.float64)
        
        # Computed Torque Replay PD Gains
        self.kp_replay = np.array(kp_replay, dtype=np.float64)
        self.kd_replay = np.array(kd_replay, dtype=np.float64)
        
        # Extract joint limits for the arm
        self.joint_limits = []
        for i in range(self.arm_dof):
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")
            rng = model.jnt_range[jnt_id]
            self.joint_limits.append(np.copy(rng))
        self.joint_limits = np.array(self.joint_limits) # shape: (6, 2)
        
        # Soft boundary cushion margin (radians) & stiffness
        self.limit_margin = 0.12  # ~7 degrees buffer zone
        self.k_limit = 80.0       # Soft bumper spring stiffness
        self.d_limit = 10.0       # Soft bumper damping

    def compute_zero_g_torque(self):
        """
        Calculates exact gravity + Coriolis cancellation torque with virtual damping
        and soft joint-limit repulsion springs.
        Returns:
            tau: (6,) ndarray of motor torques
        """
        # 1. Analytic bias torque (Coriolis + Centrifugal + Gravity) from MuJoCo RNE
        # qfrc_bias = C(q, qdot)*qdot + g(q)
        qfrc_bias = np.copy(self.data.qfrc_bias[:self.arm_dof])
        
        # 2. Virtual viscous damping: - D_v * q_dot
        qvel = self.data.qvel[:self.arm_dof]
        tau_damping = - self.damping * qvel
        
        # 3. Soft joint-limit protective cushion
        qpos = self.data.qpos[:self.arm_dof]
        tau_limit = np.zeros(self.arm_dof)
        
        for i in range(self.arm_dof):
            q_min, q_max = self.joint_limits[i]
            # Near upper limit
            if qpos[i] > q_max - self.limit_margin:
                penetration = qpos[i] - (q_max - self.limit_margin)
                tau_limit[i] -= (self.k_limit * penetration + self.d_limit * max(0.0, qvel[i]))
            # Near lower limit
            elif qpos[i] < q_min + self.limit_margin:
                penetration = (q_min + self.limit_margin) - qpos[i]
                tau_limit[i] += (self.k_limit * penetration - self.d_limit * min(0.0, qvel[i]))

        # Net command torque
        tau = qfrc_bias + tau_damping + tau_limit
        return tau

    def compute_replay_torque(self, q_ref, q_dot_ref=None):
        """
        High-precision Computed Torque Control (Feedback Linearization):
        tau = qfrc_bias + Kp * (q_ref - q) + Kd * (q_dot_ref - q_dot)
        """
        qfrc_bias = np.copy(self.data.qfrc_bias[:self.arm_dof])
        qpos = self.data.qpos[:self.arm_dof]
        qvel = self.data.qvel[:self.arm_dof]
        
        if q_dot_ref is None:
            q_dot_ref = np.zeros(self.arm_dof)
            
        pos_err = q_ref - qpos
        vel_err = q_dot_ref - qvel
        
        tau_fb = self.kp_replay * pos_err + self.kd_replay * vel_err
        tau = qfrc_bias + tau_fb
        return tau


class TrajectoryBuffer:
    """
    Manages recording, storage, smoothing, and interpolation of human-taught demonstrations.
    """
    def __init__(self, sample_dt=0.02):
        self.sample_dt = sample_dt
        self.reset()

    def reset(self):
        """Clear all recorded data."""
        self.timestamps = []
        self.qpos_list = []
        self.qvel_list = []
        self.ee_pos_list = []
        self.gripper_states = []
        self.start_time = None

    def record_sample(self, sim_time, qpos, qvel, ee_pos, gripper_state):
        """Record a single state snapshot."""
        if self.start_time is None:
            self.start_time = sim_time
            
        t_rel = sim_time - self.start_time
        self.timestamps.append(t_rel)
        self.qpos_list.append(np.copy(qpos))
        self.qvel_list.append(np.copy(qvel))
        self.ee_pos_list.append(np.copy(ee_pos))
        self.gripper_states.append(gripper_state)

    def is_empty(self):
        return len(self.timestamps) == 0

    def get_duration(self):
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def get_trajectory_arrays(self):
        """Return NumPy arrays of recorded data."""
        return (
            np.array(self.timestamps),
            np.array(self.qpos_list),
            np.array(self.qvel_list),
            np.array(self.ee_pos_list),
            np.array(self.gripper_states)
        )

    def sample_at(self, t):
        """
        Interpolates trajectory at specified relative time t.
        Returns (q_interp, qvel_interp, ee_pos_interp, gripper_state).
        """
        if len(self.timestamps) == 0:
            return None, None, None, None
            
        times = np.array(self.timestamps)
        t_clamped = np.clip(t, times[0], times[-1])
        
        # Fast binary search index
        idx = np.searchsorted(times, t_clamped)
        if idx == 0:
            return self.qpos_list[0], self.qvel_list[0], self.ee_pos_list[0], self.gripper_states[0]
        if idx >= len(times):
            return self.qpos_list[-1], self.qvel_list[-1], self.ee_pos_list[-1], self.gripper_states[-1]
            
        # Linear blend between idx-1 and idx
        t0, t1 = times[idx - 1], times[idx]
        alpha = (t_clamped - t0) / max(1e-6, (t1 - t0))
        
        q = (1.0 - alpha) * self.qpos_list[idx - 1] + alpha * self.qpos_list[idx]
        qvel = (1.0 - alpha) * self.qvel_list[idx - 1] + alpha * self.qvel_list[idx]
        ee_pos = (1.0 - alpha) * self.ee_pos_list[idx - 1] + alpha * self.ee_pos_list[idx]
        gripper = self.gripper_states[idx if alpha >= 0.5 else idx - 1]
        
        return q, qvel, ee_pos, gripper
