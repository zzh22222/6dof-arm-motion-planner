@echo off
chcp 65001 >nul
title 6-DOF Cartesian Impedance Assembly & Compliance Simulation
cd /d "%~dp0"
echo ========================================================
echo  6自由度机械臂：笛卡尔阻抗控制与顺应装配演示
echo  [1] 顺应自适应对中插装演示 (自动对中，100%%到底)
echo  [2] 完美零偏心直插演示 (垂直下潜，100%%到底)
echo  [3] 刚性卡阻反面对比演示 (刚性硬碰卡死在孔口)
echo  [4] 3D 虚拟弹性场交互游玩 (Ctrl+右键拖动体验弹力)
echo  [+] / [-] 实时动态增大 / 减小弹簧刚度
echo  提示: 启动后 1.2 秒内将自动为您演示【模式 1: 顺应装配】
echo ========================================================
python impedance_assembly.py
pause
