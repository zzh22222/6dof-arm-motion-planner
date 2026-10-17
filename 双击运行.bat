@echo off
chcp 65001 >nul
title 6-DOF Manipulator 3D Simulation
echo ========================================================
echo  正在启动 6-DOF 机械臂自主避障抓取 3D 仿真系统...
echo ========================================================
cd /d "%~dp0"
python main.py
pause
