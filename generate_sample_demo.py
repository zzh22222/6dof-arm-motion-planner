"""
Generates an exemplar demonstration trajectory for the Zero-G Lead-Through Teaching System.
Demonstrates:
1. Descend from Home posture to target cube
2. Grasp cube (Gripper Close)
3. Arc up and over the obstacle barrier
4. Lower into goal basket
5. Release cube (Gripper Open)
6. Retract to comfortable hover
"""

import numpy as np
import mujoco

from core.kinematics import ArmKinematics
from core.trajectory import PolynomialTrajectory
from core.dynamics import TrajectoryBuffer

def main():
    m = mujoco.MjModel.from_xml_path("config/robot_env_torque.xml")
    d = mujoco.MjData(m)
    kin = ArmKinematics(m, d)
    buf = TrajectoryBuffer(sample_dt=0.02)

    # Postures
    # 0. Rest
    q_home = np.array([0.0, 0.35, 0.95, 0.0, 0.65, 0.0])
    
    # 1. Pre-grasp above cube
    # Cube is at [0.45, -0.15, 0.442]
    target_pos = np.array([0.45, -0.15, 0.45])
    downward_rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    q_grasp, success = kin.solve_ik(target_pos, downward_rot, q_init=q_home)
    if not success:
        # Fallback heuristic joint values
        q_grasp = np.array([-0.32, 0.72, 0.95, 0.0, 0.68, -0.32])

    q_pre_grasp = np.copy(q_grasp)
    q_pre_grasp[1] -= 0.15

    # 2. Lift above obstacle
    q_lift = np.copy(q_pre_grasp)
    q_lift[1] -= 0.15

    # 3. Over basket
    # Basket at [0.45, 0.22, 0.43]
    basket_pos = np.array([0.45, 0.22, 0.47])
    q_place, success = kin.solve_ik(basket_pos, downward_rot, q_init=q_home)
    if not success:
        q_place = np.array([0.45, 0.70, 0.98, 0.0, 0.65, 0.45])

    q_pre_place = np.copy(q_place)
    q_pre_place[1] -= 0.15

    # Segments: (q_start, q_end, duration, gripper_open)
    segments = [
        (q_home, q_pre_grasp, 1.2, True),
        (q_pre_grasp, q_grasp, 0.8, True),
        (q_grasp, q_grasp, 0.5, False),      # Pause to grasp
        (q_grasp, q_lift, 0.9, False),        # Lift
        (q_lift, q_pre_place, 1.4, False),    # Traverse over obstacle
        (q_pre_place, q_place, 0.8, False),   # Lower into basket
        (q_place, q_place, 0.5, True),        # Release
        (q_place, q_pre_place, 0.8, True),    # Retract up
    ]

    current_time = 0.0
    dt = 0.02
    
    for q_s, q_e, dur, grip in segments:
        num_pts = int(dur / dt)
        t_arr, pos_arr, vel_arr, _ = PolynomialTrajectory.quintic(
            q_s, q_e, duration=dur, num_points=num_pts
        )
        for i in range(num_pts):
            q_val = pos_arr[i]
            v_val = vel_arr[i]
            # Compute end-effector position
            ee_p, _ = kin.forward_kinematics(q_val)
            buf.record_sample(current_time + t_arr[i], q_val, v_val, ee_p, grip)
        current_time += dur

    times, qpos, qvel, ee_pos, gripper = buf.get_trajectory_arrays()
    np.savez_compressed(
        "demo_trajectory.npz",
        times=times,
        qpos=qpos,
        qvel=qvel,
        ee_pos=ee_pos,
        gripper=gripper
    )
    print(f"Generated sample demonstration: {len(times)} points, {times[-1]:.2f} seconds.")

if __name__ == "__main__":
    main()
