"""
Interactive Zero-Gravity Lead-Through Teaching & Trajectory Replay System.

Features:
1. Dynamic Zero-G Floating: Analytical gravity & Coriolis cancellation (qfrc_bias) + virtual joint damping (Dv).
2. Lead-Through Mouse Dragging: Ctrl + Right Mouse Drag directly moves the 6-DOF arm through 3D space with zero apparent weight.
3. Demonstration Recording: Press [T] to toggle recording of human demonstrations at 50Hz.
4. Autonomous Replay: Press [P] to execute the demonstrated trajectory with smooth lead-in and computed torque control.
5. Interactive Gripper: Press [G] to grasp/release objects during teaching (gripper actions are recorded and replayed!).
6. Trajectory Persistence: Press [S] to save and [L] to load taught trajectories.
"""

import sys
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from core.kinematics import ArmKinematics
from core.dynamics import DynamicsController, TrajectoryBuffer
from core.trajectory import PolynomialTrajectory
from simulation.visualizer import TrajectoryVisualizer

# Force UTF-8 encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Rest home posture
REST_QPOS = np.array([0.0, 0.35, 0.95, 0.0, 0.65, 0.0], dtype=np.float64)

def print_banner():
    banner = """
================================================================================
   6-DOF Robot Manipulator: Zero-Gravity Lead-Through Teaching & Replay
   Dynamics:    Analytical Gravity & Coriolis Cancellation: tau = g(q) - Dv*qdot
   Apparent G:  0.000 m/s^2 (Weightless Hovering in Mid-Air)
   Controller:  High-Gain Computed Torque Feedback Linearization
================================================================================
Interactive Controls:
  [ Mouse Controls ]
    * Ctrl + Right Mouse Drag:  Drag robot arm / end-effector freely in 3D space!
    * Left Mouse Drag:          Rotate / Orbit camera perspective
    * Right Mouse Drag:         Translate / Pan camera
    * Scroll Wheel:             Zoom in / out
    * Double Left Click:        Focus camera on selected robot link

  [ Keyboard Shortcuts ]
    * [T] - Toggle TEACH / RECORD (Start / Stop recording human demonstration)
    * [P] - PLAY / REPLAY recorded trajectory (Smooth lead-in + exact replay)
    * [G] - Toggle GRIPPER Open / Close (Recorded during teaching!)
    * [R] - RESET / HOME arm smoothly to rest posture
    * [C] - CLEAR recorded trajectory buffer & visual trail
    * [S] - SAVE recorded demonstration to 'demo_trajectory.npz'
    * [L] - LOAD demonstration from 'demo_trajectory.npz'
    * [H] - Print help and controls
    * [Space] - Pause / Resume physics simulation
================================================================================
"""
    print(banner)

class ZeroGTeachingApp:
    def __init__(self, xml_path="config/robot_env_torque.xml"):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.kinematics = ArmKinematics(self.model, self.data)
        self.dynamics = DynamicsController(self.model, self.data)
        self.buffer = TrajectoryBuffer(sample_dt=0.02)
        self.visualizer = TrajectoryVisualizer(max_trail_points=250)
        
        # State machine: "FLOAT", "TEACH", "LEAD_IN", "REPLAY", "HOMING"
        self.state = "FLOAT"
        
        # Gripper state: True = open, False = closed
        self.gripper_open = True
        
        # Motion planning for lead-in and homing
        self.motion_plan = None
        self.motion_step = 0
        self.motion_total_steps = 0
        
        # Replay timing
        self.replay_start_time = 0.0
        self.replay_duration = 0.0
        
        # Performance sampling
        self.last_sample_time = 0.0
        self.sample_period = 0.02 # 50Hz demonstration sampling
        
        # Actuator IDs
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_right_finger")
        
        # Set arm initially to rest posture
        self.reset_to_rest(settle=True)

        # Auto-load pre-recorded demonstration if present
        if os.path.exists("demo_trajectory.npz"):
            self.load_trajectory("demo_trajectory.npz")

    def reset_to_rest(self, settle=True):
        """Place arm directly at rest configuration."""
        self.data.qpos[:6] = REST_QPOS
        self.data.qvel[:6] = 0.0
        self.set_gripper(open_state=True)
        mujoco.mj_forward(self.model, self.data)
        if settle:
            for _ in range(50):
                self.data.ctrl[:6] = self.dynamics.compute_zero_g_torque()
                mujoco.mj_step(self.model, self.data)

    def set_gripper(self, open_state=True):
        """Set parallel gripper fingers."""
        self.gripper_open = open_state
        if open_state:
            self.data.ctrl[self.left_finger_id] = 0.0
            self.data.ctrl[self.right_finger_id] = 0.0
        else:
            self.data.ctrl[self.left_finger_id] = -0.022
            self.data.ctrl[self.right_finger_id] = 0.022

    def toggle_gripper(self):
        """Toggle gripper state."""
        self.set_gripper(not self.gripper_open)
        status = "OPEN (Released)" if self.gripper_open else "CLOSED (Grasping)"
        print(f"[GRIPPER] Switched to: {status}")

    def toggle_teaching(self):
        """Toggle recording mode."""
        if self.state == "TEACH":
            # Stop recording
            self.state = "FLOAT"
            self.visualizer.set_mode("float")
            dur = self.buffer.get_duration()
            count = len(self.buffer.timestamps)
            print(f"\n[TEACH] Recording STOPPED! Captured {count} waypoints ({dur:.2f} seconds).")
            print("[TEACH] Ready to replay! Press [P] to execute replay demonstration.")
        else:
            # Start recording
            self.buffer.reset()
            self.visualizer.clear()
            self.visualizer.set_mode("teach")
            self.state = "TEACH"
            print("\n[TEACH] Recording STARTED! Hold [Ctrl + Right Mouse] to drag arm...")
            print("[TEACH] Press [G] to grasp/release objects during demonstration.")
            print("[TEACH] Press [T] again when finished.")

    def start_replay(self):
        """Initiate trajectory replay with smooth lead-in."""
        if self.buffer.is_empty():
            print("\n[REPLAY ERROR] No trajectory recorded! Press [T] to record first, or [L] to load.")
            return

        demo_times, demo_q, demo_v, demo_ee, demo_grip = self.buffer.get_trajectory_arrays()
        q_start = demo_q[0]
        q_current = np.copy(self.data.qpos[:6])
        
        start_dist = np.linalg.norm(q_current - q_start)
        
        if start_dist > 0.03:
            # Smooth Lead-in: Generate Quintic trajectory to return arm to start point
            duration = max(1.0, min(3.0, start_dist * 1.5))
            dt = self.model.opt.timestep
            num_pts = int(duration / dt)
            _, positions, velocities, _ = PolynomialTrajectory.quintic(
                q_current, q_start, duration=duration, num_points=num_pts
            )
            self.motion_plan = (positions, velocities)
            self.motion_step = 0
            self.motion_total_steps = num_pts
            self.state = "LEAD_IN"
            self.visualizer.set_mode("replay")
            print(f"\n[REPLAY] Smooth Lead-in: Returning arm to demonstration start point ({duration:.1f}s)...")
        else:
            self._begin_trajectory_execution()

    def _begin_trajectory_execution(self):
        """Begin executing recorded trajectory."""
        self.state = "REPLAY"
        self.visualizer.set_mode("replay")
        self.replay_start_time = self.data.time
        self.replay_duration = self.buffer.get_duration()
        print(f"[REPLAY] Executing recorded demonstration ({self.replay_duration:.2f}s) via Computed Torque Control...")

    def start_homing(self):
        """Smoothly return arm to rest home posture."""
        q_current = np.copy(self.data.qpos[:6])
        dist = np.linalg.norm(q_current - REST_QPOS)
        duration = max(1.0, min(3.0, dist * 1.2))
        dt = self.model.opt.timestep
        num_pts = int(duration / dt)
        _, positions, velocities, _ = PolynomialTrajectory.quintic(
            q_current, REST_QPOS, duration=duration, num_points=num_pts
        )
        self.motion_plan = (positions, velocities)
        self.motion_step = 0
        self.motion_total_steps = num_pts
        self.state = "HOMING"
        self.set_gripper(open_state=True)
        print(f"\n[HOMING] Returning arm to home posture ({duration:.1f}s)...")

    def save_trajectory(self, filename="demo_trajectory.npz"):
        """Save recorded trajectory buffer to file."""
        if self.buffer.is_empty():
            print("\n[SAVE ERROR] Buffer is empty! Nothing to save.")
            return
        times, qpos, qvel, ee_pos, gripper = self.buffer.get_trajectory_arrays()
        np.savez_compressed(
            filename,
            times=times,
            qpos=qpos,
            qvel=qvel,
            ee_pos=ee_pos,
            gripper=gripper
        )
        print(f"\n[SAVE] Trajectory successfully saved to: {os.path.abspath(filename)} ({len(times)} pts)")

    def load_trajectory(self, filename="demo_trajectory.npz"):
        """Load trajectory buffer from file."""
        if not os.path.exists(filename):
            print(f"\n[LOAD ERROR] File '{filename}' not found.")
            return
        data = np.load(filename)
        self.buffer.reset()
        self.buffer.timestamps = list(data["times"])
        self.buffer.qpos_list = list(data["qpos"])
        self.buffer.qvel_list = list(data["qvel"])
        self.buffer.ee_pos_list = list(data["ee_pos"])
        self.buffer.gripper_states = list(data["gripper"])
        self.buffer.start_time = 0.0
        
        # Populate visualizer with loaded trajectory
        self.visualizer.clear()
        for pt in self.buffer.ee_pos_list[::2]:
            self.visualizer.add_point(pt)
            
        dur = self.buffer.get_duration()
        print(f"\n[LOAD] Loaded trajectory from '{filename}' ({len(self.buffer.timestamps)} pts, {dur:.2f}s). Press [P] to replay.")

    def on_key(self, key):
        """Key callback from MuJoCo viewer."""
        try:
            char = chr(key).upper()
        except Exception:
            return

        if char == 'T':
            self.toggle_teaching()
        elif char == 'P':
            self.start_replay()
        elif char == 'G':
            self.toggle_gripper()
        elif char == 'R':
            self.start_homing()
        elif char == 'C':
            self.buffer.reset()
            self.visualizer.clear()
            self.state = "FLOAT"
            self.visualizer.set_mode("float")
            print("\n[CLEAR] Trajectory buffer & visual ribbon cleared.")
        elif char == 'S':
            self.save_trajectory()
        elif char == 'L':
            self.load_trajectory()
        elif char == 'H':
            print_banner()

    def step(self, viewer=None):
        """Advance physics simulation by one sub-step."""
        # 1. Apply user mouse perturbations from 3D viewer
        if viewer is not None:
            if viewer.perturb.select > 0 and viewer.perturb.localmass <= 0:
                viewer.perturb.localmass = max(2.5, self.model.body_mass[viewer.perturb.select] * 4.0)
                viewer.perturb.scale = 1.0

            self.data.xfrc_applied[:] = 0.0
            mujoco.mjv_applyPerturbPose(self.model, self.data, viewer.perturb, 0)
            mujoco.mjv_applyPerturbForce(self.model, self.data, viewer.perturb)

        # 2. Current end-effector position
        ee_pos = self.kinematics.get_ee_position()
        self.visualizer.add_point(ee_pos)

        # State dispatch
        if self.state == "FLOAT":
            # Pure Zero-Gravity Hovering + Viscous Damping
            tau = self.dynamics.compute_zero_g_torque()
            self.data.ctrl[:6] = tau
            self.set_gripper(self.gripper_open)

        elif self.state == "TEACH":
            # Hover + Sample demonstration at 50Hz
            tau = self.dynamics.compute_zero_g_torque()
            self.data.ctrl[:6] = tau
            self.set_gripper(self.gripper_open)
            
            if self.data.time - self.last_sample_time >= self.sample_period:
                self.buffer.record_sample(
                    self.data.time,
                    self.data.qpos[:6],
                    self.data.qvel[:6],
                    ee_pos,
                    self.gripper_open
                )
                self.last_sample_time = self.data.time

        elif self.state == "LEAD_IN":
            # Smooth interpolation towards start posture
            positions, velocities = self.motion_plan
            if self.motion_step < self.motion_total_steps:
                q_ref = positions[self.motion_step]
                v_ref = velocities[self.motion_step]
                tau = self.dynamics.compute_replay_torque(q_ref, v_ref)
                self.data.ctrl[:6] = tau
                self.motion_step += 1
            else:
                # Finished lead-in, start demonstration playback
                self._begin_trajectory_execution()

        elif self.state == "REPLAY":
            # Execute recorded demonstration
            t_rel = self.data.time - self.replay_start_time
            if t_rel <= self.replay_duration:
                q_ref, v_ref, _, grip_state = self.buffer.sample_at(t_rel)
                if q_ref is not None:
                    tau = self.dynamics.compute_replay_torque(q_ref, v_ref)
                    self.data.ctrl[:6] = tau
                    self.set_gripper(grip_state)
            else:
                # Replay finished, seamlessly lock into zero-G hover
                self.state = "FLOAT"
                self.visualizer.set_mode("float")
                print("\n[REPLAY COMPLETE] Demonstration finished! Robot arm locked in mid-air hover.")

        elif self.state == "HOMING":
            positions, velocities = self.motion_plan
            if self.motion_step < self.motion_total_steps:
                q_ref = positions[self.motion_step]
                v_ref = velocities[self.motion_step]
                tau = self.dynamics.compute_replay_torque(q_ref, v_ref)
                self.data.ctrl[:6] = tau
                self.motion_step += 1
            else:
                self.state = "FLOAT"
                self.visualizer.set_mode("float")
                print("\n[HOMING COMPLETE] Arm reached home rest posture.")

        # Advance physics
        mujoco.mj_step(self.model, self.data)

def main():
    parser = argparse.ArgumentParser(description="Zero-Gravity Lead-Through Teaching System")
    parser.add_argument("--xml", type=str, default="config/robot_env_torque.xml", help="Path to robot XML model")
    args = parser.parse_args()

    print_banner()

    app = ZeroGTeachingApp(xml_path=args.xml)

    print("[INIT] Launching interactive MuJoCo 3D Viewer...")
    with mujoco.viewer.launch_passive(app.model, app.data, key_callback=app.on_key) as viewer:
        # Perspective camera
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -22.0
        viewer.cam.distance = 1.70
        viewer.cam.lookat[:] = [0.38, 0.04, 0.52]

        # Auto-select gripper end-effector handle by default so user can drag immediately!
        gripper_body_id = mujoco.mj_name2id(app.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        if gripper_body_id >= 0:
            viewer.perturb.select = gripper_body_id
            viewer.perturb.active = 0
            viewer.perturb.localmass = 3.0
            viewer.perturb.scale = 1.0

        print("\n[SYSTEM READY] Zero-Gravity hover active! Arm is weightless.")
        print("[HINT] Hold [Ctrl + Right Mouse Button] to drag the arm in 3D.")
        print("[HINT] Double-click any link to select and drag a different joint.")
        print("[HINT] Press [T] to start recording a trajectory!\n")

        # Main rendering loop (~60-100 FPS with 5 physics substeps at dt=0.002)
        while viewer.is_running():
            for _ in range(5):
                app.step(viewer)
                
            app.visualizer.render(viewer)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
