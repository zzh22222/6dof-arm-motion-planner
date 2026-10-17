"""
Peg-in-Hole Precision Assembly Task using Cartesian Impedance Control.

Key Demonstration Modes:
1. Compliant Impedance Insertion (顺应自适应插装):
   Anisotropic compliance (soft XY, stiff Z) allows contact forces to passively 
   guide and self-center the peg into the socket, achieving 100% full insertion depth.
   
2. Rigid High-Stiffness Baseline (刚性硬碰硬卡死):
   High stiffness rejects lateral deflection; misalignment creates massive reaction 
   forces and wedging/jamming on the rim, stalling insertion at the entrance.
   
3. Interactive 3D Virtual Spring (3D 虚拟弹性场游玩):
   Peg tip is tethered to a 3D equilibrium anchor point via an adjustable virtual 
   spring-damper field. Users can drag the peg with mouse, feeling elastic restoring forces!
"""

import time
import numpy as np
import mujoco

from core.impedance import CartesianImpedanceController
from core.kinematics import ArmKinematics

class PegInHoleTask:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Ensure kinematics state is computed before reading sites
        mujoco.mj_forward(self.model, self.data)
        
        self.kin = ArmKinematics(model, data, ee_site_name="peg_tip")
        self.ctrl = CartesianImpedanceController(model, data, site_name="peg_tip")
        
        # Hole sites
        self.s_entry = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_entry_site")
        self.s_bottom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_bottom_site")
        
        # Nominal hole coordinates
        self.hole_entry_pos = np.copy(self.data.site_xpos[self.s_entry])   # [0.46, 0.0, 0.488]
        self.hole_bottom_pos = np.copy(self.data.site_xpos[self.s_bottom]) # [0.46, 0.0, 0.440]
        
        # Downward tool orientation
        self.R_downward = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
        
        # Current operating mode
        self.mode = "COMPLIANT" # "COMPLIANT", "RIGID", "INTERACTIVE"
        
        # Task metrics
        self.insertion_depth = 0.0
        self.max_contact_force = 0.0
        self.lateral_deflection = 0.0

    def reset(self, offset_x=0.0035, offset_y=0.0025):
        """
        Reset arm to pre-assembly approach pose directly above the hole with intentional tolerance offset.
        offset_x, offset_y: intentional position error to simulate manufacturing tolerances.
        """
        mujoco.mj_forward(self.model, self.data)
        self.hole_entry_pos = np.copy(self.data.site_xpos[self.s_entry])
        self.hole_bottom_pos = np.copy(self.data.site_xpos[self.s_bottom])

        # Approach pose 5.5cm directly above hole with controlled lateral misalignment
        approach_pos = np.array([
            self.hole_entry_pos[0] + offset_x,
            self.hole_entry_pos[1] + offset_y,
            self.hole_entry_pos[2] + 0.055
        ])
        
        q_init = np.array([0.0, 0.45, 0.95, 0.0, 0.65, 0.0])
        q_sol, success = self.kin.solve_ik(approach_pos, self.R_downward, q_init=q_init)
        
        if success:
            self.kin.set_joint_positions(q_sol)
        else:
            self.data.qpos[:6] = q_init
            
        self.data.qvel[:6] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # Reset metrics
        self.insertion_depth = 0.0
        self.max_contact_force = 0.0
        self.lateral_deflection = 0.0

    def compute_metrics(self):
        """Calculate insertion depth percentage and lateral deflection."""
        peg_pos, _ = self.ctrl.get_ee_pose()
        total_depth = self.hole_entry_pos[2] - self.hole_bottom_pos[2] # ~0.048m
        current_travel = self.hole_entry_pos[2] - peg_pos[2]
        self.insertion_depth = np.clip(current_travel / max(1e-4, total_depth), 0.0, 1.0)
        
        # Lateral distance to hole centerline
        self.lateral_deflection = np.linalg.norm(peg_pos[:2] - self.hole_entry_pos[:2]) * 1000.0 # mm
        return self.insertion_depth, self.lateral_deflection
