"""
Trajectory Comparison & Analysis Script.
Compares Cubic (3rd-Order) vs Quintic (5th-Order) Polynomial Trajectories.
Generates publication-quality plots of Position, Velocity, Acceleration, and Jerk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from core.trajectory import PolynomialTrajectory

def generate_trajectory_plots(save_path="trajectory_analysis.png"):
    # Simulated joint motion from 0.0 to 1.5 rad over 2.0 seconds
    q0 = 0.0
    qf = 1.5
    duration = 2.0
    num_points = 200
    
    # 1. Cubic Trajectory
    t_c, pos_c, vel_c, acc_c = PolynomialTrajectory.cubic(
        q0, qf, v0=0.0, vf=0.0, duration=duration, num_points=num_points
    )
    jerk_c = np.gradient(acc_c, t_c, axis=0)
    
    # 2. Quintic Trajectory
    t_q, pos_q, vel_q, acc_q = PolynomialTrajectory.quintic(
        q0, qf, v0=0.0, vf=0.0, a0=0.0, af=0.0, duration=duration, num_points=num_points
    )
    jerk_q = np.gradient(acc_q, t_q, axis=0)
    
    # Create Figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    # 1. Position
    axes[0].plot(t_c, pos_c, 'r--', label='Cubic Polynomial (3rd-order)', linewidth=2)
    axes[0].plot(t_q, pos_q, 'b-', label='Quintic Polynomial (5th-order)', linewidth=2.5)
    axes[0].set_ylabel('Position [rad]', fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].set_title('Robotic Manipulator Trajectory Profiles: Cubic vs. Quintic', fontsize=13, fontweight='bold')
    
    # 2. Velocity
    axes[1].plot(t_c, vel_c, 'r--', label='Cubic Velocity', linewidth=2)
    axes[1].plot(t_q, vel_q, 'b-', label='Quintic Velocity', linewidth=2.5)
    axes[1].set_ylabel('Velocity [rad/s]', fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right', frameon=True)
    
    # 3. Acceleration
    axes[2].plot(t_c, acc_c, 'r--', label='Cubic Acceleration (Discontinuous jump at boundaries!)', linewidth=2)
    axes[2].plot(t_q, acc_q, 'b-', label='Quintic Acceleration (Smooth, 0 at boundaries)', linewidth=2.5)
    axes[2].set_ylabel('Acceleration [rad/s²]', fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper right', frameon=True)
    
    # 4. Jerk (Derivative of acceleration)
    axes[3].plot(t_c, jerk_c, 'r--', label='Cubic Jerk', linewidth=2)
    axes[3].plot(t_q, jerk_q, 'b-', label='Quintic Jerk (Bounded, Low Motor Vibration)', linewidth=2.5)
    axes[3].set_ylabel('Jerk [rad/s³]', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time [seconds]', fontsize=11, fontweight='bold')
    axes[3].grid(True, linestyle='--', alpha=0.6)
    axes[3].legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Trajectory comparison plot saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    generate_trajectory_plots()
