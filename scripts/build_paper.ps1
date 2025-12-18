@echo off
setlocal
cd /d C:\Users\erezs\OneDrive\913C~1\RBD-CA
.\\.venv\\Scripts\\python analysis\\aggregate_sweep.py
.\\.venv\\Scripts\\python paper\\build_report.py
echo ----------------------------------------
echo PAPER PREVIEW (paper_v1.md first 30 lines)
echo ----------------------------------------
powershell -NoProfile -Command "Get-Content -Path 'paper/paper_v1.md' -TotalCount 30"
echo ----------------------------------------
echo RESULTS TABLE (results_table.md first 30 lines)
echo ----------------------------------------
powershell -NoProfile -Command "Get-Content -Path 'paper/results_table.md' -TotalCount 30"
echo DONE build_paper

