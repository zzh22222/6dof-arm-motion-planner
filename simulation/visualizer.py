"""
3D Trajectory Visualizer for MuJoCo Viewer.
Draws dynamic real-time glowing trajectory ribbons and planned waypoints.
"""

import numpy as np
import mujoco

class TrajectoryVisualizer:
    def __init__(self, max_trail_points=160):
        self.max_trail_points = max_trail_points
        self.trail_history = []
        self.planned_waypoints = []
        self._identity_mat = np.eye(3).flatten()
        self.mode = "float" # "float", "teach", "replay"
        self.status_pos = np.array([0.1, 0.0, 0.95]) # Indicator position above base

    def set_mode(self, mode):
        """Set current visualization theme: 'float', 'teach', 'replay'."""
        self.mode = mode

    def set_planned_waypoints(self, waypoints):
        """Store planned RRT waypoints to render as golden landmark spheres."""
        self.planned_waypoints = [np.array(wp, dtype=float) for wp in waypoints]

    def add_point(self, point):
        """Append a new end-effector position to the visual trail history."""
        pt = np.array(point, dtype=float)
        # Avoid storing identical consecutive points
        if len(self.trail_history) == 0 or np.linalg.norm(self.trail_history[-1] - pt) > 0.003:
            self.trail_history.append(pt)
            if len(self.trail_history) > self.max_trail_points:
                self.trail_history.pop(0)

    def clear(self):
        """Clear visual trail history."""
        self.trail_history.clear()
        self.planned_waypoints.clear()

    def render(self, viewer):
        """Inject visual geoms directly into MuJoCo viewer user_scn."""
        if viewer is None or not viewer.is_running():
            return
            
        user_scn = viewer.user_scn
        geom_idx = 0
        max_geoms = len(user_scn.geoms)

        # 1. Render Status Indicator Bead (hovering in scene)
        if geom_idx < max_geoms:
            if self.mode == "teach":
                status_color = [0.1, 0.95, 0.2, 0.9] # Bright emerald green (recording)
                status_size = [0.02, 0.0, 0.0]
            elif self.mode == "replay":
                status_color = [1.0, 0.8, 0.05, 0.9] # Radiant gold (replaying)
                status_size = [0.02, 0.0, 0.0]
            else:
                status_color = [0.1, 0.7, 1.0, 0.5] # Subtle cyan (zero-g floating)
                status_size = [0.012, 0.0, 0.0]

            mujoco.mjv_initGeom(
                user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=status_size,
                pos=self.status_pos,
                mat=self._identity_mat,
                rgba=status_color
            )
            geom_idx += 1

        # 2. Render Planned Waypoint Landmarks (Gold spheres)
        for wp in self.planned_waypoints:
            if geom_idx >= max_geoms:
                break
            mujoco.mjv_initGeom(
                user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.012, 0.0, 0.0],
                pos=wp,
                mat=self._identity_mat,
                rgba=[1.0, 0.75, 0.1, 0.85]  # Bright Gold
            )
            geom_idx += 1

        # 3. Render Trajectory Trail
        num_pts = len(self.trail_history)
        for i, pt in enumerate(self.trail_history):
            if geom_idx >= max_geoms:
                break
            fraction = (i + 1) / max(1, num_pts)
            
            if self.mode == "teach":
                # Emerald to Lime Green gradient
                r = float(0.05 + 0.15 * fraction)
                g = float(0.7 + 0.3 * fraction)
                b = float(0.2 + 0.3 * (1.0 - fraction))
                alpha = float(0.3 + 0.65 * fraction)
                radius = float(0.005 + 0.004 * fraction)
            elif self.mode == "replay":
                # Orange to Radiant Gold gradient
                r = float(1.0)
                g = float(0.4 + 0.55 * fraction)
                b = float(0.05)
                alpha = float(0.35 + 0.6 * fraction)
                radius = float(0.005 + 0.005 * fraction)
            else:
                # Electric Cyan gradient
                r = float(0.1 + 0.3 * (1.0 - fraction))
                g = float(0.5 + 0.45 * fraction)
                b = float(0.9 + 0.1 * fraction)
                alpha = float(0.2 + 0.65 * fraction)
                radius = float(0.004 + 0.004 * fraction)
            
            mujoco.mjv_initGeom(
                user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0.0, 0.0],
                pos=pt,
                mat=self._identity_mat,
                rgba=[r, g, b, alpha]
            )
            geom_idx += 1

        user_scn.ngeom = geom_idx

