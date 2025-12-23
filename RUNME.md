# Time as Alignment Cost – Reproducibility Guide

## Quick Start

### Prerequisites
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r code/requirements.txt
```

### Option A: Regenerate from existing sweep data
```bash
# If you have outputs_sweep50/ with completed runs:
python code/regenerate_all.py
```

### Option B: Run full sweep from scratch
```powershell
# Windows PowerShell (takes ~60 minutes for 500 runs)
powershell -ExecutionPolicy Bypass -File scripts/run_sweep.ps1 -seeds "0..49" -steps 300 -outRoot "outputs_sweep50"

# Then regenerate figures/tables:
python code/regenerate_all.py
```

### Compile LaTeX
```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # twice for references
```

## Package Contents

```
paper/
  main.tex          # LaTeX source
  main.pdf          # Compiled 13-page paper
  data_summary.json # Run counts, correlations, verification
  figs/             # Figures 1-5

code/
  bda.py              # CA simulation with coherence-cost
  coherence_cost.py   # S_coh, S_pred, S_total calculator
  regenerate_all.py   # Rebuild tables/figs from CSVs
  verify_stats.py     # Verify completeness & stats
  requirements.txt    # Python dependencies

data/
  sweep_summary_by_mode.csv  # Aggregated statistics
  sweep_manifest.txt         # All 500 runs with status
```

## Figure 5 Statistics (from CSVs)

| Boundary | All Modes (n) | r | Persistence Only (n) | r |
|----------|---------------|---|---------------------|---|
| zero     | 250           | 0.89 | 100              | 0.70 |
| wrap     | 250           | 0.89 | 100              | 0.78 |

## Verification Checklist

- [x] 500/500 runs completed (2 boundaries × 5 modes × 50 seeds)
- [x] Tables 1-2 regenerated from sweep data
- [x] Figures 1-5 present in paper
- [x] No appendix in document
- [x] S_total = S_coh + S_pred (λ=1)
- [x] Â = exp(-S_total) in Figure 5
