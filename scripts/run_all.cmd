@echo off
setlocal
cd /d C:\Users\erezs\OneDrive\913C~1\RBD-CA
.\.venv\Scripts\python -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_sweep.ps1
.\.venv\Scripts\python analysis\aggregate_sweep.py
echo DONE run_all

