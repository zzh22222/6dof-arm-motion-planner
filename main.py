"""
Main Entry Point: 6-DOF Industrial Manipulator Vision-to-Action Motion Planning & Grasping.

Features:
- Native GPU-accelerated MuJoCo 3D Interactive Viewer
- Eye-to-Hand Overhead Industrial Vision Perception (OpenCV Color Segmentation + Pinhole Calibration)
- RRT-Connect Obstacle Avoidance Motion Planning with Physical Safety Barrier
- Real-time 3D Trajectory Ribbon & Waypoint Markers
- DLS Inverse Kinematics with Null-Space Posture Optimization
- Quintic Polynomial C2 Smooth Velocity and Acceleration Profiles
"""

import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from simulation.robot import RobotArmEnv
from simulation.visualizer import TrajectoryVisualizer
from tasks.pick_and_place import PickAndPlaceTask

# Force UTF-8 encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_banner(obstacle_avoidance=True, randomize=False):
    mode = "RRT-Connect Obstacle Avoidance" if obstacle_avoidance else "Baseline Direct Trajectory (Crash Expected)"
    cube_mode = "Randomized Table Position" if randomize else "Standard Benchmark Position"
    banner = f"""
========================================================================
   6-DOF Industrial Robotic Arm - Vision-to-Action Autonomous Grasping
   Perception:  Overhead Camera + OpenCV (Hand-Eye Inversion: ~1.5mm Err)
   Planning:    {mode}
   Target Pose: {cube_mode}
   Kinematics:  DLS Inverse Kinematics + Null-Space Posture Bias
   Trajectory:  Quintic (5th-Order) C2 Polynomial Motion Planning
   Visuals:     Real-time 3D Trajectory Ribbon & Waypoint Markers
   Physics:     DeepMind MuJoCo Engine (500Hz Implicit Integration)
========================================================================
3D Viewer Controls:
  - Left Mouse Drag:   Orbit / Rotate Camera View
  - Right Mouse Drag:  Pan / Translate Camera
  - Scroll Wheel:      Zoom In / Out
  - Double Left Click: Focus Camera on Clicked Object
  - Spacebar:          Pause / Resume Physics Simulation
========================================================================
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="6-DOF Arm Vision-to-Action Motion Planning")
    parser.add_argument("--baseline", action="store_true", 
                        help="Run baseline straight trajectory without obstacle avoidance (demonstrates collision)")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomly perturb target cube position on the table to test dynamic vision detection")
    parser.add_argument("--no-vision", action="store_true",
                        help="Disable camera perception and use simulator ground-truth coordinates")
    args = parser.parse_args()
    
    use_avoidance = not args.baseline
    use_vision = not args.no_vision
    print_banner(obstacle_avoidance=use_avoidance, randomize=args.randomize)
    
    # 1. Initialize Robot, Task, and Visualizer
    print("[INIT] Loading 6-DOF robot model and physics world...")
    env = RobotArmEnv(xml_path="config/robot_env.xml")
    if args.randomize:
        print("[INIT] Randomizing cube placement on table...")
        env.reset(randomize_cube=True)
        
    visualizer = TrajectoryVisualizer(max_trail_points=160)
    task = PickAndPlaceTask(env)
    
    print("[INIT] Launching MuJoCo 3D Viewer...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Set cinematic default perspective angle
        viewer.cam.azimuth = 142.0
        viewer.cam.elevation = -24.0
        viewer.cam.distance = 1.75
        viewer.cam.lookat[:] = [0.38, 0.04, 0.52]
        
        # Warmup and settle scene
        for _ in range(40):
            env.step(1)
            visualizer.render(viewer)
            viewer.sync()
            time.sleep(0.01)
            
        print("[READY] Physics & Vision stable. Demo beginning in 2 seconds...")
        for i in range(2, 0, -1):
            if not viewer.is_running():
                return
            print(f"Starting in {i}...")
            for _ in range(30):
                env.step(1)
                visualizer.render(viewer)
                viewer.sync()
                time.sleep(0.033)

        # 2. Execute Autonomous Task with Vision and 3D Visualizer
        task.run_cycle(viewer=viewer, visualizer=visualizer, 
                       obstacle_avoidance=use_avoidance, use_vision=use_vision)
        
        # 3. Keep viewer running for user exploration
        print("\n[IDLE] Pick & Place finished! 3D Trajectory trail is locked in air.")
        print("[IDLE] Use your mouse to rotate and inspect the 3D trajectory curve.")
        print("[IDLE] Detection HUD saved to: vision_detection.png")
        print("[IDLE] Close the 3D window when finished.")
        
        while viewer.is_running():
            env.step(1)
            visualizer.render(viewer)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
