"""
Automated Cinematic Demo Recorder for 6-DOF Manipulator.
Renders high-definition offscreen frames from the cinematic camera and exports:
1. demo_avoidance.gif (Optimized for GitHub README)
2. demo_avoidance.mp4 (High-res video)
Supports --baseline flag to record collision crash comparison.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import mujoco

from simulation.robot import RobotArmEnv
from tasks.pick_and_place import PickAndPlaceTask

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class OffscreenVideoRecorder:
    def __init__(self, model, data, camera_name="cinematic_cam", width=640, height=480):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.frames = []
        self.step_counter = 0
        self.record_interval = 6  # Record 1 frame every 6 physics steps (~40 FPS)

    def step_and_capture(self, num_steps=1):
        """Step physics and conditionally capture a video frame."""
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
            self.step_counter += 1
            if self.step_counter % self.record_interval == 0:
                self.renderer.update_scene(self.data, camera=self.camera_name)
                frame = self.renderer.render()
                self.frames.append(frame)

    def save_outputs(self, prefix="demo_avoidance"):
        mp4_path = f"{prefix}.mp4"
        gif_path = f"{prefix}.gif"
        
        num_frames = len(self.frames)
        print(f"\n[RECORD] Captured {num_frames} frames. Processing outputs...")
        
        if num_frames == 0:
            print("[WARN] No frames captured.")
            return
            
        # 1. Export MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(mp4_path, fourcc, 30.0, (self.width, self.height))
        for frame in self.frames:
            # MuJoCo outputs RGB, OpenCV expects BGR
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[SUCCESS] Exported MP4 Video: {os.path.abspath(mp4_path)}")
        
        # 2. Export Optimized GIF for GitHub README (Downsampled to 20 FPS, 480x360 for compact size)
        print("[RECORD] Compressing GIF for GitHub README embedding...")
        gif_frames = []
        # Downsample frames by 2 to keep GIF lightweight (<10MB)
        for i in range(0, num_frames, 2):
            img = Image.fromarray(self.frames[i])
            img_resized = img.resize((480, 360), Image.Resampling.LANCZOS)
            gif_frames.append(img_resized)
            
        if gif_frames:
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                optimize=True,
                duration=50,  # 20 FPS (50ms per frame)
                loop=0
            )
            gif_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            print(f"[SUCCESS] Exported GitHub GIF:  {os.path.abspath(gif_path)} ({gif_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Record cinematic demo of 6-DOF Manipulator")
    parser.add_argument("--baseline", action="store_true", help="Record baseline collision crash instead of avoidance")
    args = parser.parse_args()
    
    prefix = "demo_crash" if args.baseline else "demo_avoidance"
    avoidance = not args.baseline
    
    print("=======================================================")
    print(f"🎬 [RECORD] Starting Automated Cinematic Video Capture")
    print(f"   Target Output: {prefix}.mp4 & {prefix}.gif")
    print(f"   Mode:          {'RRT Obstacle Avoidance' if avoidance else 'Baseline Crash'}")
    print("=======================================================")
    
    env = RobotArmEnv(xml_path="config/robot_env.xml")
    recorder = OffscreenVideoRecorder(env.model, env.data, camera_name="cinematic_cam")
    
    # Replace env.step with recorder.step_and_capture
    env.step = recorder.step_and_capture
    
    task = PickAndPlaceTask(env)
    task.run_cycle(viewer=None, visualizer=None, obstacle_avoidance=avoidance, use_vision=True)
    
    # Extra pause at end
    recorder.step_and_capture(num_steps=150)
    
    # Save files
    recorder.save_outputs(prefix=prefix)
    print("\n[COMPLETE] Recording finished successfully!")

if __name__ == "__main__":
    main()
