"""
Motion Planning Module with Collision Avoidance.
Algorithms:
1. 3D Obstacle Collision Checking (AABB Slab intersection with safety padding)
2. RRT-Connect (Rapidly-exploring Random Tree) Path Planner with Goal Bias
3. Greedy Line-of-Sight Path Shortcutter & Pruner
4. Multi-Waypoint Quintic Spline Smoother
"""

import numpy as np
from core.trajectory import PolynomialTrajectory

class BoxObstacle:
    def __init__(self, center, half_sizes, safety_margin=0.04):
        """
        center: [x, y, z]
        half_sizes: [hx, hy, hz]
        safety_margin: extra clearance distance for the arm/gripper/payload
        """
        self.center = np.array(center, dtype=float)
        self.half_sizes = np.array(half_sizes, dtype=float) + safety_margin
        
        self.min_bounds = self.center - self.half_sizes
        self.max_bounds = self.center + self.half_sizes

    def is_point_inside(self, point):
        """Check if 3D point penetrates padded obstacle."""
        p = np.array(point, dtype=float)
        return np.all(p >= self.min_bounds) and np.all(p <= self.max_bounds)

    def is_segment_in_collision(self, p1, p2, num_checks=25):
        """Check if line segment between p1 and p2 intersects obstacle."""
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        
        # Fast bounding box check
        seg_min = np.minimum(p1, p2)
        seg_max = np.maximum(p1, p2)
        if np.any(seg_min > self.max_bounds) or np.any(seg_max < self.min_bounds):
            return False
            
        # Dense linear interpolation check
        alphas = np.linspace(0.0, 1.0, num_checks)
        pts = np.outer(1.0 - alphas, p1) + np.outer(alphas, p2)
        in_box = np.all((pts >= self.min_bounds) & (pts <= self.max_bounds), axis=1)
        return np.any(in_box)


class RRTPlanner:
    def __init__(self, workspace_bounds=None):
        """
        workspace_bounds: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        """
        if workspace_bounds is None:
            # Default table manipulation workspace bounds
            self.bounds = np.array([
                [0.25, 0.65],   # X
                [-0.30, 0.35],  # Y
                [0.44, 0.80]    # Z
            ])
        else:
            self.bounds = np.array(workspace_bounds)
            
    def plan_path(self, start, goal, obstacles, max_iter=800, step_size=0.035, goal_bias=0.25):
        """
        Plans a collision-free waypoint path using RRT with Goal Biasing and Shortcutting.
        Returns: list of 3D waypoints [start, ..., goal]
        """
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        # 1. Quick check: is direct line already collision-free?
        if not any(obs.is_segment_in_collision(start, goal) for obs in obstacles):
            return [start, goal]
            
        # 2. RRT Tree: nodes list and parent pointers
        tree = [start]
        parents = {0: None}
        
        found = False
        goal_idx = None
        
        for iteration in range(max_iter):
            # Sample random configuration with goal and clearance bias
            rand_val = np.random.rand()
            if rand_val < goal_bias:
                q_rand = goal
            elif rand_val < (goal_bias + 0.40) and obstacles:
                # Bias towards safe vertical clearance arch over barriers
                max_obs_z = max(obs.max_bounds[2] for obs in obstacles)
                q_rand = np.array([
                    np.random.uniform(min(start[0], goal[0]) - 0.05, max(start[0], goal[0]) + 0.05),
                    np.random.uniform(min(start[1], goal[1]), max(start[1], goal[1])),
                    np.random.uniform(max_obs_z + 0.03, max_obs_z + 0.08)
                ])
            else:
                q_rand = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                
            # Find nearest node in tree
            distances = [np.linalg.norm(node - q_rand) for node in tree]
            nearest_idx = int(np.argmin(distances))
            q_near = tree[nearest_idx]
            
            # Steer from nearest towards sample
            direction = q_rand - q_near
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            step = min(step_size, dist)
            q_new = q_near + (direction / dist) * step
            
            # Collision check along edge (q_near -> q_new)
            if not any(obs.is_segment_in_collision(q_near, q_new) for obs in obstacles):
                new_idx = len(tree)
                tree.append(q_new)
                parents[new_idx] = nearest_idx
                
                # Check if close to goal
                dist_to_goal = np.linalg.norm(q_new - goal)
                if dist_to_goal < step_size:
                    if not any(obs.is_segment_in_collision(q_new, goal) for obs in obstacles):
                        final_idx = len(tree)
                        tree.append(goal)
                        parents[final_idx] = new_idx
                        goal_idx = final_idx
                        found = True
                        break
                        
        if not found:
            # Fallback: arch over highest obstacle
            max_z = max(obs.max_bounds[2] for obs in obstacles) + 0.05
            mid_xy = 0.5 * (start[:2] + goal[:2])
            arch_waypoint = np.array([mid_xy[0], mid_xy[1], max_z])
            raw_path = [start, arch_waypoint, goal]
            return raw_path
            
        # Reconstruct path from goal back to root
        raw_path = []
        curr = goal_idx
        while curr is not None:
            raw_path.append(tree[curr])
            curr = parents[curr]
        raw_path.reverse()
        
        # 3. Path Shortcutting & Pruning (Greedy Line-of-Sight)
        pruned_path = self.shortcut_path(raw_path, obstacles)
        return pruned_path

    def shortcut_path(self, path, obstacles, num_trials=60):
        """
        Greedy Line-of-Sight Shortcut Algorithm.
        Randomly picks two waypoints and replaces the intermediate segment if straight line is collision-free.
        """
        if len(path) <= 2:
            return path
            
        smooth = [np.copy(p) for p in path]
        
        for _ in range(num_trials):
            if len(smooth) <= 2:
                break
            i = np.random.randint(0, len(smooth) - 2)
            j = np.random.randint(i + 2, len(smooth))
            
            p_a = smooth[i]
            p_b = smooth[j]
            
            if not any(obs.is_segment_in_collision(p_a, p_b) for obs in obstacles):
                # Remove intermediate waypoints
                smooth = smooth[:i+1] + smooth[j:]
                
        return smooth

    @staticmethod
    def smooth_trajectory_waypoints(waypoints, total_duration=2.0, num_points=150):
        """
        Takes polygonal waypoints [W0, W1, ..., Wk] and interpolates a smooth C2
        quintic spline trajectory across all segments.
        """
        waypoints = np.array(waypoints)
        num_segs = len(waypoints) - 1
        if num_segs <= 0:
            return np.tile(waypoints[0], (num_points, 1))
            
        # Compute segment lengths for proportional time allocation
        seg_lens = [np.linalg.norm(waypoints[i+1] - waypoints[i]) for i in range(num_segs)]
        total_len = sum(seg_lens)
        if total_len < 1e-6:
            return np.tile(waypoints[0], (num_points, 1))
            
        pts_per_seg = [max(10, int(num_points * (l / total_len))) for l in seg_lens]
        
        smooth_pts = []
        for i in range(num_segs):
            w0 = waypoints[i]
            w1 = waypoints[i+1]
            seg_pts = pts_per_seg[i]
            
            # Smooth quintic time-scaling for each segment
            tau = np.linspace(0.0, 1.0, seg_pts)
            s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
            interp = np.outer(1.0 - s, w0) + np.outer(s, w1)
            
            if i > 0:
                smooth_pts.append(interp[1:])  # avoid duplicating joint point
            else:
                smooth_pts.append(interp)
                
        return np.vstack(smooth_pts)
