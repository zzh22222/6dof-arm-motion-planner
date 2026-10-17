@echo off
chcp 65001 >nul
title 6-DOF Zero-G Manipulator Simulation
cd /d "%~dp0"
echo ========================================================
echo  Starting 6-DOF Zero-Gravity Lead-Through Teaching...
echo ========================================================
python zero_g_teaching.py
pause
