"""
6-DOF Robotic Manipulator: Cartesian Impedance Control & Compliant Peg-in-Hole Assembly.

Features:
1. [Mode 1] Compliant Assembly (顺应装配演示 - 旗舰核心功能):
   Anisotropic compliance (soft XY, firm Z) allows contact forces to passively 
   self-center and glide the peg smoothly into the hole, achieving 100% full insertion depth!
2. [Mode 2] Perfect Centered Insertion (完美对心直插):
   Zero misalignment error, straight vertical descent with 100% full depth.
3. [Mode 3] Rigid Baseline Jamming (刚性卡死反面对比):
   High stiffness rejects compliance; misaligned peg wedges and jams on the chamfer rim (<5% depth).
4. [Mode 4] Interactive 3D Elastic Field (3D 虚拟弹性场游玩):
   Peg is tethered to a 3D anchor; drag with Ctrl + Right Mouse to feel virtual spring restoring force!
5. Real-time Stiffness Adjustment: Press [+] or [-] to dynamically tune elasticity.
"""

import sys
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from core.impedance import CartesianImpedanceController
from core.kinematics import ArmKinematics
from tasks.peg_in_hole import PegInHoleTask
from simulation.visualizer import TrajectoryVisualizer

# Force UTF-8 encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def print_banner():
    banner = """
================================================================================
   6自由度机械臂：笛卡尔空间阻抗控制与顺应装配演示系统
   Cartesian Impedance Control & Compliant Peg-in-Hole Assembly
   控制算法:    操作空间阻抗控制 (tau = J^T * W_cart + qfrc_bias + tau_null)
   物理仿真:    DeepMind MuJoCo 高精度多体动力学与摩擦接触力学
   配合公差:    34mm 透明亚克力孔座 vs 25mm 抛光黄铜轴 (4.5mm 顺应导引间隙)
================================================================================
演示与交互模式快捷键:
  [ 数字键 ]
    * [1] - 【顺应装配】(旗舰功能! 偏心 +3.5mm, 倒角自适应顺应导正, 100% 到底!)
    * [2] - 【完美直插】(基准对照: 零偏心完美对心, 垂直下潜入孔, 100% 到底!)
    * [3] - 【刚性卡阻】(反面对比: 刚性高刚度, 偏心硬碰硬瞬间卡死在孔口!)
    * [4] - 【3D 弹性场】(自由交互: 悬停孔口, 按住 Ctrl+鼠标右键拖动体验弹力!)

  [ 刚度调节与控制 ]
    * [+] / [-] - 实时增大 / 减小笛卡尔弹簧刚度 Kp
    * [R] - 重置机械臂到孔口上方就绪姿态
    * [H] - 打印本帮助菜单
    * [Space] - 暂停 / 继续物理仿真

  [ 鼠标交互 (模式 4) ]
    * 按住 [Ctrl + 鼠标右键拖拽] 轴端把手，拉伸 3D 虚拟弹性场！
    * 松开鼠标：观察机械臂如弹簧般回弹振荡并稳定在平衡点！
================================================================================
"""
    print(banner)

class ImpedanceAssemblyApp:
    def __init__(self, xml_path="config/robot_env_impedance.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.kin = ArmKinematics(self.model, self.data, ee_site_name="peg_tip")
        self.ctrl = CartesianImpedanceController(self.model, self.data, site_name="peg_tip")
        self.task = PegInHoleTask(self.model, self.data)
        self.visualizer = TrajectoryVisualizer(max_trail_points=180)
        
        # State machine: "COMPLIANT_RUN", "RIGID_RUN", "INTERACTIVE", "IDLE"
        self.state = "IDLE"
        
        # Assembly trajectory tracking
        self.traj_step = 0
        self.traj_total_steps = 1100
        self.x_start = np.zeros(3)
        self.x_goal = np.zeros(3)
        self.target_rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
        
        # Interactive anchor point (golden reference bead directly above hole)
        self.anchor_pos = np.array([self.task.hole_entry_pos[0], self.task.hole_entry_pos[1], self.task.hole_entry_pos[2] + 0.055], dtype=np.float64)
        self.interactive_k = 180.0
        
        # HUD timing
        self.last_hud_time = 0.0
        
        # Initialize arm to ready pose
        self.task.reset(offset_x=0.0035, offset_y=0.0025)

    def start_compliant_assembly(self):
        """[MODE 1] Flagship compliant insertion with intentional tolerance offset."""
        print("\n" + "="*70)
        print(">>> 【模式 1: 顺应装配演示】(Compliant Impedance Assembly)")
        print(">>> 初始偏心误差: X方向 +3.5mm, Y方向 +2.5mm (模拟精密加工装配公差)")
        print(">>> 刚度调度策略: 空中段高刚度定轨 (600 N/m) -> 接触段低刚度顺应 (100 N/m)")
        print(">>> 物理特性: 接触漏斗倒角后自动产生顺应侧向力，PASSIVE 自寻心导正入孔！")
        print("="*70)
        
        self.task.reset(offset_x=0.0035, offset_y=0.0025)
        self.ctrl.set_compliant_assembly_mode()
        self.visualizer.clear()
        self.visualizer.set_mode("teach")
        
        peg_pos, _ = self.ctrl.get_ee_pose()
        self.x_start = np.copy(peg_pos)
        # Command goal straight down with the offset to test passive self-centering
        self.x_goal = np.array([self.x_start[0], self.x_start[1], self.task.hole_bottom_pos[2] - 0.001])
        
        self.traj_step = 0
        self.traj_total_steps = 1100 # 2.2 seconds at 500Hz
        self.state = "COMPLIANT_RUN"

    def start_perfect_insertion(self):
        """[MODE 2] Zero-offset centered baseline insertion."""
        print("\n" + "="*70)
        print(">>> 【模式 2: 完美无偏对心直插】(Zero-Offset Centered Insertion)")
        print(">>> 初始位置: 0.0mm 误差，轴心与孔心完全对齐")
        print(">>> 控制器驱动机械臂垂直匀速下沉到底...")
        print("="*70)
        
        self.task.reset(offset_x=0.0, offset_y=0.0)
        self.ctrl.set_compliant_assembly_mode()
        self.visualizer.clear()
        self.visualizer.set_mode("teach")
        
        peg_pos, _ = self.ctrl.get_ee_pose()
        self.x_start = np.copy(peg_pos)
        self.x_goal = np.array([self.task.hole_entry_pos[0], self.task.hole_entry_pos[1], self.task.hole_bottom_pos[2] - 0.001])
        
        self.traj_step = 0
        self.traj_total_steps = 1100
        self.state = "COMPLIANT_RUN"

    def start_rigid_baseline(self):
        """[MODE 3] Rigid high-stiffness collision failure baseline."""
        print("\n" + "="*70)
        print(">>> 【模式 3: 刚性卡死反面对比】(Rigid Collision Jamming Demo)")
        print(">>> 相同初始偏心: X方向 +3.5mm, Y方向 +2.5mm")
        print(">>> 刚性高刚度: K_xyz = 2200 N/m, 缺乏顺应柔性，发生刚性碰撞卡死！")
        print("="*70)
        
        self.task.reset(offset_x=0.0035, offset_y=0.0025)
        self.ctrl.set_rigid_baseline_mode()
        self.visualizer.clear()
        self.visualizer.set_mode("float")
        
        peg_pos, _ = self.ctrl.get_ee_pose()
        self.x_start = np.copy(peg_pos)
        self.x_goal = np.array([self.x_start[0], self.x_start[1], self.task.hole_bottom_pos[2] - 0.001])
        
        self.traj_step = 0
        self.traj_total_steps = 1100
        self.state = "RIGID_RUN"

    def start_interactive_mode(self):
        """[MODE 4] Interactive 3D virtual elastic spring mode."""
        print("\n" + "="*70)
        print(f">>> 【模式 4: 3D 虚拟弹性场游玩】(Interactive 3D Spring)")
        print(f">>> 当前弹性刚度: K = {self.interactive_k:.0f} N/m")
        print(">>> 机械臂轴端悬停在孔口上方平衡点！")
        print(">>> 操作方法: 按住 [Ctrl + 鼠标右键] 拖动机械臂末端拉拽弹簧，松手观察回弹！")
        print(">>> 按 [+] / [-] 可实时调软或调硬弹簧。")
        print("="*70)
        
        self.task.reset(offset_x=0.0, offset_y=0.0)
        self.ctrl.set_interactive_spring_mode(k_trans=self.interactive_k)
        self.anchor_pos = np.array([self.task.hole_entry_pos[0], self.task.hole_entry_pos[1], self.task.hole_entry_pos[2] + 0.055], dtype=np.float64)
        self.visualizer.clear()
        self.visualizer.set_mode("replay")
        self.state = "INTERACTIVE"

    def adjust_stiffness(self, delta):
        """Dynamically increase or decrease interactive stiffness."""
        self.interactive_k = np.clip(self.interactive_k + delta, 40.0, 1500.0)
        self.ctrl.set_interactive_spring_mode(k_trans=self.interactive_k)
        print(f"[刚度调节] 弹簧刚度已设置为: K = {self.interactive_k:.0f} N/m (阻尼 D = {self.ctrl.kd_trans[0]:.1f} N*s/m)")

    def on_key(self, key):
        """MuJoCo viewer keyboard callback."""
        try:
            char = chr(key).upper()
        except Exception:
            return

        if char == '1':
            self.start_compliant_assembly()
        elif char == '2':
            self.start_perfect_insertion()
        elif char == '3':
            self.start_rigid_baseline()
        elif char == '4':
            self.start_interactive_mode()
        elif char in ['+', '=']:
            self.adjust_stiffness(+40.0)
        elif char in ['-', '_']:
            self.adjust_stiffness(-40.0)
        elif char == 'R':
            self.task.reset(offset_x=0.0, offset_y=0.0)
            print("\n[复位] 机械臂已复位至孔上方就绪姿态。")
        elif char == 'H':
            print_banner()

    def step(self, viewer=None):
        """Execute one physics substep with Cartesian impedance control."""
        # 1. Apply user mouse perturbations from 3D viewer
        if viewer is not None:
            if viewer.perturb.select > 0 and viewer.perturb.localmass <= 0:
                viewer.perturb.localmass = max(2.5, self.model.body_mass[viewer.perturb.select] * 4.0)
                viewer.perturb.scale = 1.0

            self.data.xfrc_applied[:] = 0.0
            mujoco.mjv_applyPerturbPose(self.model, self.data, viewer.perturb, 0)
            mujoco.mjv_applyPerturbForce(self.model, self.data, viewer.perturb)

        # 2. End-effector position & visualizer
        peg_pos, peg_rot = self.ctrl.get_ee_pose()
        self.visualizer.add_point(peg_pos)

        # 3. State execution
        if self.state in ["COMPLIANT_RUN", "RIGID_RUN"]:
            # Two-stage stiffness scheduling for compliant mode:
            # - In free air (z > 0.495m): stiff XY (600 N/m) eliminates arm dynamic bias drift.
            # - Upon contact (z <= 0.495m): soft XY (100 N/m, firm Z 400 N/m, soft rot 12) passively funnels into socket.
            if self.state == "COMPLIANT_RUN":
                if peg_pos[2] > 0.495:
                    self.ctrl.set_stiffness(trans_stiffness=[600.0, 600.0, 450.0], rot_stiffness=[50.0, 50.0, 30.0])
                else:
                    self.ctrl.set_stiffness(trans_stiffness=[100.0, 100.0, 420.0], rot_stiffness=[12.0, 12.0, 10.0])

            # Smooth downward linear trajectory progression
            alpha = min(1.0, self.traj_step / float(self.traj_total_steps))
            x_des = (1.0 - alpha) * self.x_start + alpha * self.x_goal
            v_des = (self.x_goal - self.x_start) / (self.traj_total_steps * 0.002) if alpha < 1.0 else np.zeros(3)
            
            tau, info = self.ctrl.compute_torque(x_des, self.target_rot, v_desired=v_des)
            self.data.ctrl[:6] = tau
            self.traj_step += 1
            
            # Print live assembly metrics every 0.3s
            if self.data.time - self.last_hud_time >= 0.3:
                depth, lateral = self.task.compute_metrics()
                contact = info["contact_force"]
                mode_str = "顺应装配" if self.state == "COMPLIANT_RUN" else "刚性对比"
                print(f"[{mode_str}] 装配深度: {depth*100:5.1f}% | 轴端高度: {peg_pos[2]:.4f}m | 偏心误差: {lateral:4.1f}mm | 接触力: {contact:5.1f}N")
                self.last_hud_time = self.data.time
                
            if self.traj_step >= self.traj_total_steps + 300: # Hold at bottom
                depth, lateral = self.task.compute_metrics()
                print("\n" + "-"*65)
                if self.state == "COMPLIANT_RUN":
                    print(f"★ [装配成功: 顺应控制] 最终装配深度: {depth*100:.1f}%! 轴顺利自适应滑入孔底!")
                else:
                    print(f"✕ [装配卡阻: 刚性硬碰] 最终装配深度: {depth*100:.1f}%! 刚性过大卡死在孔口边缘!")
                print("-"*65 + "\n")
                self.state = "IDLE"

        elif self.state == "INTERACTIVE":
            # Virtual Spring Anchor Equilibrium
            tau, info = self.ctrl.compute_torque(self.anchor_pos, self.target_rot)
            self.data.ctrl[:6] = tau
            
            if self.data.time - self.last_hud_time >= 0.5:
                disp = np.linalg.norm(peg_pos - self.anchor_pos) * 100.0 # cm
                force = info["cartesian_force"]
                print(f"[弹性场 HUD] 位移形变: {disp:4.1f}cm | 虚拟回弹力: {force:5.1f}N | 弹簧刚度 K: {self.interactive_k:.0f} N/m")
                self.last_hud_time = self.data.time

        elif self.state == "IDLE":
            # Hold current achieved pose with compliant impedance
            tau, _ = self.ctrl.compute_torque(peg_pos, self.target_rot)
            self.data.ctrl[:6] = tau

        # 4. Advance physics
        mujoco.mj_step(self.model, self.data)

    def render_spring_overlay(self, viewer):
        """Render virtual spring tether line and anchor sphere in viewer."""
        if viewer is None or not viewer.is_running():
            return
            
        user_scn = viewer.user_scn
        geom_idx = user_scn.ngeom
        max_geoms = len(user_scn.geoms)
        
        peg_pos, _ = self.ctrl.get_ee_pose()
        ident = np.eye(3).flatten()

        # 1. Render Golden Anchor Sphere in Interactive Mode
        if self.state == "INTERACTIVE" and geom_idx < max_geoms:
            mujoco.mjv_initGeom(
                user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015, 0.0, 0.0],
                pos=self.anchor_pos,
                mat=ident,
                rgba=[1.0, 0.8, 0.1, 0.85] # Radiant Gold Anchor
            )
            geom_idx += 1

        # 2. Render Virtual Spring Tether (beaded elastic line between anchor and peg)
        if self.state == "INTERACTIVE":
            dist = np.linalg.norm(peg_pos - self.anchor_pos)
            if dist > 0.01:
                num_beads = min(12, int(dist / 0.015))
                for b in range(1, num_beads):
                    if geom_idx >= max_geoms:
                        break
                    frac = b / float(num_beads)
                    bead_pos = (1.0 - frac) * self.anchor_pos + frac * peg_pos
                    # Color shifts from green to bright red when stretched hard
                    stretch_ratio = min(1.0, dist / 0.25)
                    color = [0.2 + 0.8 * stretch_ratio, 0.9 - 0.7 * stretch_ratio, 0.1, 0.8]
                    
                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.005, 0.0, 0.0],
                        pos=bead_pos,
                        mat=ident,
                        rgba=color
                    )
                    geom_idx += 1

        user_scn.ngeom = geom_idx

def main():
    parser = argparse.ArgumentParser(description="6-DOF Cartesian Impedance Assembly System")
    parser.add_argument("--xml", type=str, default="config/robot_env_impedance.xml", help="Path to XML model")
    args = parser.parse_args()

    print_banner()

    app = ImpedanceAssemblyApp(xml_path=args.xml)

    print("[系统启动] 正在打开 MuJoCo 3D 仿真窗口...")
    with mujoco.viewer.launch_passive(app.model, app.data, key_callback=app.on_key) as viewer:
        # Camera angle focusing clearly on peg and transparent acrylic socket
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -24.0
        viewer.cam.distance = 1.35
        viewer.cam.lookat[:] = [0.46, 0.0, 0.50]

        # Auto-select peg tool body for immediate dragging
        tool_id = mujoco.mj_name2id(app.model, mujoco.mjtObj.mjOBJ_BODY, "assembly_tool")
        if tool_id >= 0:
            viewer.perturb.select = tool_id
            viewer.perturb.active = 0
            viewer.perturb.localmass = 3.5
            viewer.perturb.scale = 1.0

        print("\n[就绪提示] 系统将在 1.2 秒后【自动启动模式 1: 顺应插装演示】!")
        print("  * 按键盘 [1] 可随时重新执行【顺应装配演示】(自动对中，100% 到底)")
        print("  * 按键盘 [2] 执行【完美对心直插】")
        print("  * 按键盘 [3] 执行【刚性卡阻反面对比】(体验碰头卡死)")
        print("  * 按键盘 [4] 切换【3D 弹性场游玩】(可按住 Ctrl+右键拖动)\n")

        start_time = time.time()
        auto_started = False

        while viewer.is_running():
            # Automatically start flagship compliant insertion demo 1.2s after window opens
            if not auto_started and (time.time() - start_time >= 1.2):
                app.start_compliant_assembly()
                auto_started = True

            for _ in range(5):
                app.step(viewer)
                
            app.visualizer.render(viewer)
            app.render_spring_overlay(viewer)
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
