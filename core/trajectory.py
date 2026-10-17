"""
Trajectory Planning Module for Robotic Manipulators.
Provides:
1. Cubic (3rd-order) polynomial joint interpolation
2. Quintic (5th-order) polynomial joint interpolation (zero jerk / smooth boundary accel)
3. Cartesian pose trajectory interpolation (Linear + Slerp)
"""

import numpy as np

class PolynomialTrajectory:
    @staticmethod
    def cubic(q0, qf, v0=0.0, vf=0.0, duration=1.0, num_points=100):
        """
        Cubic (3rd-order) polynomial trajectory generator.
        Boundary conditions:
        q(0) = q0, q(T) = qf, q_dot(0) = v0, q_dot(T) = vf
        """
        T = max(duration, 1e-4)
        t = np.linspace(0, T, num_points)
        
        # Coefficients
        a0 = q0
        a1 = v0
        a2 = (3 * (qf - q0) - (2 * v0 + vf) * T) / (T ** 2)
        a3 = (-2 * (qf - q0) + (v0 + vf) * T) / (T ** 3)
        
        positions = np.outer(np.ones_like(t), a0) + np.outer(t, a1) + np.outer(t**2, a2) + np.outer(t**3, a3)
        velocities = np.outer(np.ones_like(t), a1) + np.outer(2*t, a2) + np.outer(3*t**2, a3)
        accelerations = np.outer(np.ones_like(t), 2*a2) + np.outer(6*t, a3)
        
        return t, positions, velocities, accelerations

    @staticmethod
    def quintic(q0, qf, v0=0.0, vf=0.0, a0=0.0, af=0.0, duration=1.0, num_points=100):
        """
        Quintic (5th-order) polynomial trajectory generator.
        Boundary conditions:
        q(0) = q0, q(T) = qf
        q_dot(0) = v0, q_dot(T) = vf
        q_ddot(0) = a0, q_ddot(T) = af
        Provides smooth C2 continuity with zero terminal acceleration.
        """
        T = max(duration, 1e-4)
        t = np.linspace(0, T, num_points)
        
        # Delta terms
        h = qf - q0
        
        # Coefficients
        c0 = q0
        c1 = v0
        c2 = 0.5 * a0
        c3 = (20 * h - (8 * vf + 12 * v0) * T - (3 * a0 - af) * (T ** 2)) / (2 * (T ** 3))
        c4 = (-30 * h + (14 * vf + 16 * v0) * T + (3 * a0 - 2 * af) * (T ** 2)) / (2 * (T ** 4))
        c5 = (12 * h - 6 * (vf + v0) * T + (af - a0) * (T ** 2)) / (2 * (T ** 5))
        
        positions = (np.outer(np.ones_like(t), c0) + 
                     np.outer(t, c1) + 
                     np.outer(t**2, c2) + 
                     np.outer(t**3, c3) + 
                     np.outer(t**4, c4) + 
                     np.outer(t**5, c5))
                     
        velocities = (np.outer(np.ones_like(t), c1) + 
                      np.outer(2*t, c2) + 
                      np.outer(3*t**2, c3) + 
                      np.outer(4*t**3, c4) + 
                      np.outer(5*t**4, c5))
                      
        accelerations = (np.outer(np.ones_like(t), 2*c2) + 
                         np.outer(6*t, c3) + 
                         np.outer(12*t**2, c4) + 
                         np.outer(20*t**3, c5))
                         
        return t, positions, velocities, accelerations


class CartesianTrajectory:
    """
    Interpolates 3D Cartesian positions and orientations (quaternion slerp).
    """
    @staticmethod
    def linear_position(start_pos, end_pos, duration=1.0, num_points=100, smooth=True):
        """
        Interpolates 3D positions with either linear or quintic-time-scaling.
        """
        start_pos = np.array(start_pos, dtype=float)
        end_pos = np.array(end_pos, dtype=float)
        t = np.linspace(0, duration, num_points)
        
        if smooth:
            # Quintic time scaling s(t) in [0, 1]
            tau = t / duration
            s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
        else:
            s = t / duration
            
        positions = np.outer(1 - s, start_pos) + np.outer(s, end_pos)
        return t, positions

    @staticmethod
    def slerp(q0, q1, num_points=100):
        """
        Spherical linear interpolation (SLERP) between two unit quaternions [x, y, z, w].
        """
        q0 = np.array(q0, dtype=float)
        q1 = np.array(q1, dtype=float)
        q0 /= np.linalg.norm(q0)
        q1 /= np.linalg.norm(q1)
        
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
            
        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        
        if np.abs(theta_0) < 1e-6:
            return np.tile(q0, (num_points, 1))
            
        sin_theta_0 = np.sin(theta_0)
        t_vals = np.linspace(0.0, 1.0, num_points)
        
        quaternions = []
        for t in t_vals:
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.cos(theta_t) - dot * sin_theta_t / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            q_interp = s0 * q0 + s1 * q1
            quaternions.append(q_interp / np.linalg.norm(q_interp))
            
        return np.array(quaternions)
