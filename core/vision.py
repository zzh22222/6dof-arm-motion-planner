"""
Computer Vision & Eye-to-Hand Perception Module.
Uses OpenCV for real-time target recognition, color segmentation, and 3D pose reconstruction.
"""

import numpy as np
import cv2
import mujoco

class ObjectVisionDetector:
    def __init__(self, model, data, camera_name="overhead_cam", img_width=640, img_height=480):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        self.width = img_width
        self.height = img_height
        self.renderer = mujoco.Renderer(model, height=img_height, width=img_width)
        
        # Camera Intrinsics
        fovy_rad = np.deg2rad(model.cam_fovy[self.cam_id])
        self.fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx = self.fy  # Square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # Table top height reference (cube center is ~0.44m in world frame)
        self.target_z_world = 0.440

    def capture_rgb(self):
        """Render RGB image from the overhead camera."""
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb_img = self.renderer.render()
        return rgb_img

    def detect_red_cube_pose(self, save_debug_img="vision_detection.png"):
        """
        Detects red cube in camera feed and reconstructs 3D world coordinates.
        Returns:
            detected_3d_pos: np.array([x, y, z]) in robot base / world coordinates
            annotated_img: RGB image with bounding boxes, crosshairs, and text overlay
        """
        rgb = self.capture_rgb()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Red color thresholding (two intervals in HSV space)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 100, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological opening/closing to remove sensor noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated_img = np.copy(rgb)
        
        if not contours:
            return None, annotated_img
            
        # Select largest red contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 80:  # Noise threshold
            return None, annotated_img
            
        # Compute image moments for sub-pixel centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None, annotated_img
            
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        
        # Bounding box
        x_box, y_box, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # --- Pinhole Inverse Projection (Hand-Eye Transform) ---
        # Camera extrinsics from MuJoCo
        cam_pos = np.copy(self.data.cam_xpos[self.cam_id])
        cam_rot = np.copy(self.data.cam_xmat[self.cam_id].reshape(3, 3))
        
        # Depth along camera optical axis (-Z in MuJoCo camera local frame)
        depth = cam_pos[2] - self.target_z_world
        
        # Normalized coordinates in pinhole camera
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        # Point in camera coordinates (MuJoCo camera frame: X=right, Y=up, Z=-forward)
        # Note: image v increases downwards, so Y_cam is inverted
        P_cam = np.array([x_norm * depth, -y_norm * depth, -depth])
        
        # Transform from camera frame to world frame: P_world = R_cam * P_cam + pos_cam
        P_world = cam_rot @ P_cam + cam_pos
        detected_pos = np.array([P_world[0], P_world[1], self.target_z_world])
        
        # --- Draw HUD & Telemetry Overlays on Image ---
        cv2.rectangle(annotated_img, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)
        center_px = (int(round(u)), int(round(v)))
        cv2.drawMarker(annotated_img, center_px, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        
        # Text overlays
        hud_text1 = f"TARGET: Red Cube | Pixels: ({center_px[0]}, {center_px[1]})"
        hud_text2 = f"3D Estimated World: [{detected_pos[0]:.3f}, {detected_pos[1]:.3f}, {detected_pos[2]:.3f}] m"
        
        cv2.putText(annotated_img, hud_text1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(annotated_img, hud_text2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        if save_debug_img:
            # Convert RGB to BGR for OpenCV file writing
            cv2.imwrite(save_debug_img, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
            
        return detected_pos, annotated_img
