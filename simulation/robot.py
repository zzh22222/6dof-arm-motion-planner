"""
Robot Simulation and Execution Interface for 6-DOF Manipulator in MuJoCo.
Supports Joint Trajectories, Cartesian Trajectories, Multi-Waypoint Paths, and 3D Visualizer.
"""

import time
import numpy as np
import mujoco
from core.kinematics import ArmKinematics
from core.trajectory import PolynomialTrajectory, CartesianTrajectory

class RobotArmEnv:
    def __init__(self, xml_path="config/robot_env.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kinematics = ArmKinematics(self.model, self.data)
        
        # Actuator IDs
        self.arm_actuators = [f"act_j{i+1}" for i in range(6)]
        self.arm_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.arm_actuators]
        
        self.act_left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_left_finger")
        self.act_right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_right_finger")
        
        # Object bodies
        self.target_cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        self.goal_basket_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_basket")
        self.barrier_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_barrier")
        
        # Initialize robot at home configuration
        self.reset()

    def reset(self, randomize_cube=False):
        """Reset physics simulation and set arm to rest posture."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Optionally randomize cube location on table (Wide noticeable range)
        if randomize_cube:
            cube_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
            qpos_adr = self.model.jnt_qposadr[cube_jnt]
            self.data.qpos[qpos_adr] = np.random.uniform(0.36, 0.52)      # 16cm X-axis variation
            self.data.qpos[qpos_adr + 1] = np.random.uniform(-0.24, -0.09) # 15cm Y-axis variation
            self.data.qpos[qpos_adr + 2] = 0.445
            
        # Set joint angles to rest posture
        self.kinematics.set_joint_positions(self.kinematics.q_rest)
        for act_id, q in zip(self.arm_act_ids, self.kinematics.q_rest):
            self.data.ctrl[act_id] = q
            
        # Gripper open initially
        self.set_gripper(open_state=True)
        
        # Forward simulation to settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def get_ee_pose(self):
        """Get end-effector position and orientation matrix."""
        return self.kinematics.forward_kinematics()

    def get_cube_pos(self):
        """Get current 3D position of the target cube."""
        return np.copy(self.data.xpos[self.target_cube_id])

    def get_basket_pos(self):
        """Get current 3D position of the destination basket."""
        return np.copy(self.data.xpos[self.goal_basket_id])

    def get_barrier_info(self):
        """Get obstacle barrier position and dimensions for planner."""
        pos = np.copy(self.data.xpos[self.barrier_id])
        # [hx, hy, hz] = [0.08, 0.015, 0.11]
        half_sizes = np.array([0.08, 0.015, 0.11])
        return pos, half_sizes

    def set_gripper(self, open_state=True):
        """
        Control parallel gripper fingers.
        open_state=True: fully opened
        open_state=False: closed / grasp
        """
        if open_state:
            self.data.ctrl[self.act_left_finger_id] = 0.0
            self.data.ctrl[self.act_right_finger_id] = 0.0
        else:
            self.data.ctrl[self.act_left_finger_id] = -0.022
            self.data.ctrl[self.act_right_finger_id] = 0.022

    def step(self, num_substeps=1):
        """Advance physics simulation."""
        for _ in range(num_substeps):
            mujoco.mj_step(self.model, self.data)

    def execute_joint_trajectory(self, q_start, q_target, duration=1.5, viewer=None, visualizer=None, method="quintic"):
        """
        Execute smooth trajectory in joint space from q_start to q_target.
        Uses quintic or cubic polynomial planning.
        """
        dt = self.model.opt.timestep
        num_points = int(duration / dt)
        
        if method == "quintic":
            _, pos_traj, _, _ = PolynomialTrajectory.quintic(
                q0=q_start, qf=q_target, duration=duration, num_points=num_points
            )
        else:
            _, pos_traj, _, _ = PolynomialTrajectory.cubic(
                q0=q_start, qf=q_target, duration=duration, num_points=num_points
            )
            
        for q_step in pos_traj:
            for act_id, q_val in zip(self.arm_act_ids, q_step):
                self.data.ctrl[act_id] = q_val
            self.step(1)
            
            if visualizer is not None:
                ee_pos, _ = self.get_ee_pose()
                visualizer.add_point(ee_pos)
                
            if viewer is not None and viewer.is_running():
                if visualizer is not None:
                    visualizer.render(viewer)
                viewer.sync()
                time.sleep(dt * 0.4)

    def execute_cartesian_trajectory(self, target_pos, duration=1.5, viewer=None, visualizer=None):
        """
        Move end-effector along a smooth straight Cartesian path to target_pos.
        """
        current_pos, _ = self.get_ee_pose()
        current_q = self.kinematics.get_joint_positions()
        
        target_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        
        q_target, ok = self.kinematics.solve_ik(target_pos, target_rot=target_rot, q_init=current_q)
        if not ok:
            q_target, _ = self.kinematics.solve_ik(target_pos, target_rot=None, q_init=current_q)
            
        self.execute_joint_trajectory(current_q, q_target, duration=duration, 
                                      viewer=viewer, visualizer=visualizer, method="quintic")

    def execute_cartesian_path(self, dense_waypoints, duration=2.5, viewer=None, visualizer=None):
        """
        Execute a multi-waypoint collision-free path (from RRT planner) smoothly.
        Iteratively tracks waypoints with DLS IK.
        """
        num_pts = len(dense_waypoints)
        dt = duration / max(1, num_pts)
        current_q = self.kinematics.get_joint_positions()
        
        target_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        
        substeps = max(1, int(dt / self.model.opt.timestep))
        
        for pt in dense_waypoints:
            # Solve DLS IK for current point (using previous joint solution as warm start)
            q_sol, _ = self.kinematics.solve_ik(
                target_pos=pt, target_rot=target_rot, q_init=current_q, 
                max_iter=15, damping=0.02
            )
            current_q = q_sol
            
            for act_id, q_val in zip(self.arm_act_ids, q_sol):
                self.data.ctrl[act_id] = q_val
                
            self.step(substeps)
            
            if visualizer is not None:
                ee_pos, _ = self.get_ee_pose()
                visualizer.add_point(ee_pos)
                
            if viewer is not None and viewer.is_running():
                if visualizer is not None:
                    visualizer.render(viewer)
                viewer.sync()
                time.sleep(dt * 0.3)
