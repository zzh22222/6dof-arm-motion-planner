@echo off
chcp 65001 >nul
title 6-DOF Manipulator Baseline Collision (Crash Demo)
echo ========================================================
echo  正在启动 基准对比模式 (无避障规划 - 观察猛烈撞墙效果)...
echo ========================================================
cd /d "%~dp0"
python main.py --baseline
pause
