@echo off
chcp 65001 >nul
title 6-DOF Manipulator Vision-to-Action 3D Simulation
echo ========================================================
echo  正在启动 视觉引导随机抓取 (Vision-to-Action) 3D 仿真...
echo ========================================================
cd /d "%~dp0"
python main.py --randomize
pause
