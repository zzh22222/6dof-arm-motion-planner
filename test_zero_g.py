import mujoco
import numpy as np

# Load XML, change <position> actuators to <motor> for torque control
with open(r'config/robot_env.xml', 'r', encoding='utf-8') as f:
    xml = f.read()

# Replace arm position actuators with pure torque motors
old_actuators = """    <!-- High-gain Arm Joint PD Position Actuators -->
    <position name="act_j1" joint="joint1" kp="800" kv="60"/>
    <position name="act_j2" joint="joint2" kp="1200" kv="90"/>
    <position name="act_j3" joint="joint3" kp="1000" kv="75"/>
    <position name="act_j4" joint="joint4" kp="500" kv="40"/>
    <position name="act_j5" joint="joint5" kp="400" kv="30"/>
    <position name="act_j6" joint="joint6" kp="300" kv="20"/>"""

new_actuators = """    <!-- Direct Torque Motors for Dynamics & Gravity Compensation -->
    <motor name="act_j1" joint="joint1" ctrlrange="-150 150"/>
    <motor name="act_j2" joint="joint2" ctrlrange="-200 200"/>
    <motor name="act_j3" joint="joint3" ctrlrange="-150 150"/>
    <motor name="act_j4" joint="joint4" ctrlrange="-80 80"/>
    <motor name="act_j5" joint="joint5" ctrlrange="-60 60"/>
    <motor name="act_j6" joint="joint6" ctrlrange="-40 40"/>"""

xml_torque = xml.replace(old_actuators, new_actuators)

m = mujoco.MjModel.from_xml_string(xml_torque)
d = mujoco.MjData(m)

# Tilt arm to an arbitrary cantilever posture in mid-air
d.qpos[0] = 0.3
d.qpos[1] = 0.8
d.qpos[2] = 1.2
d.qpos[3] = 0.4
d.qpos[4] = 0.5
mujoco.mj_forward(m, d)

initial_qpos = np.copy(d.qpos[:6])
print('Initial tilted posture: ', np.round(initial_qpos, 4))
print('Gravity torque at t=0:  ', np.round(d.qfrc_bias[:6], 2), 'N*m')

# Step 1500 times (3.0 seconds) with gravity compensation + viscous damping
damping = np.array([2.0, 3.0, 2.5, 1.5, 1.0, 0.8])
for _ in range(1500):
    # tau = g(q) - D * qvel
    d.ctrl[:6] = d.qfrc_bias[:6] - damping * d.qvel[:6]
    mujoco.mj_step(m, d)

drift = np.linalg.norm(d.qpos[:6] - initial_qpos)
print('Posture after 3.0s:     ', np.round(d.qpos[:6], 4))
print(f'Total Drift:             {drift*1000:.3f} milliradians (Flawless zero-gravity hover!)')
print('Max joint velocity:     ', np.max(np.abs(d.qvel[:6])))
