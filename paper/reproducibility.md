# Reproducibility Guide

## Environment
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run sweep (current paper settings)
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_sweep.ps1 -seeds 0..9 -steps 300
```
- Modes: object_off_tracked, passive_nopersist, passive_persist, active_persist
- Boundary modes: zero, wrap
- Perturbation: on, t=150, rate=0.03, radius=10
- Outputs per run: `outputs/<tag>/metrics.csv`
- Manifest: `analysis_outputs/sweep_manifest.txt`

## Aggregate + paper tables
```
.venv\Scripts\python analysis\aggregate_sweep.py
.venv\Scripts\python paper\build_report.py
```
- Summaries: `analysis_outputs/sweep_summary_by_mode.csv`, `analysis_outputs/sweep_summary_by_seed.csv`
- Figures: `analysis_outputs/fig_identity_by_mode.png`, `fig_recovery_by_mode.png`, `fig_action_rate_by_mode.png`
- Paper table: `paper/results_table.md`

## Quick preview (optional)
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

## Inspect summaries (no PowerShell quoting pain)
```
.venv\Scripts\python - << "PY"
import pandas as pd
df = pd.read_csv("analysis_outputs/sweep_summary_by_mode.csv")
print(df.head(10))
PY
```

## Regenerate manuscript assets
- Copy figures to `paper/figures/` if not already present:
  `copy analysis_outputs\fig_identity_by_mode.png paper\figures\`
  `copy analysis_outputs\fig_recovery_by_mode.png paper\figures\`
  `copy analysis_outputs\fig_action_rate_by_mode.png paper\figures\`
- Manuscript: `paper/manuscript.md`
- Appendix: `paper/appendix_metrics.md`

