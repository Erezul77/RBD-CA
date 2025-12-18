# Recursive Boundary Dynamics in Cellular Systems: A Boundary-Driven Automaton with Observer-Dependent Identity Stabilization

## Abstract
We study a boundary-driven cellular automaton (DBC-CA) augmented with a finite observer that can track and, when active, locally stabilize a chosen entity. Boundary conditions are either zero-background or toroidal wrap, avoiding metaphysical claims of “vacuum.” We measure identity continuity (identity_score), shock recovery (core_size_delta), and intervention budgets (observer_action_rate). Multi-seed sweeps show that an active, budgeted observer maintains higher identity continuity than passive or object-only baselines across both boundary modes.

## 1. Introduction
Boundaries mediate interaction between active patterns and a background medium. We frame “observation as action”: a subject-like process that selects, tracks, and can minimally intervene to preserve an entity’s identity. We present a tractable CA model with explicit boundary modes and observer modes, and quantify identity stabilization under shocks.

## 2. Model
- Binary CA on H×W grid, Moore neighborhood.
- Distance-to-boundary thresholds a(d), b(d) set survival/birth; boundary computed per boundary-mode (zero or wrap).
- Perturbation: one-time flip shock at t=150 within a radius around attention.

## 3. Observer/Subject
- Modes: off_tracked; passive_nopersist; passive_persist; active_persist.
- Passive: segmentation + tracking (identity lock via persistence bias); no rule changes.
- Active: same tracking plus budgeted overrides (observer_budget) inside the attention window; only a subset of differing cells are applied per step.
- Tracking: core selected in window; attention drifts toward core; identity_score prioritizes persist_iou else core_jaccard_prev.

## 4. Metrics
- identity_score (persist_iou or core_jaccard_prev)
- core_size_delta (recovery after shock baseline)
- observer_action_rate (fraction of window cells overridden; active only)
- boundary_mode, observer_mode tags

## 5. Experiments
- Sweeps over boundary_mode ∈ {zero, wrap} and observer_mode ∈ {off_tracked, passive_nopersist, passive_persist, active_persist}, seeds 0–2 in latest run (extendable).
- Steps=300, perturb_t=150, perturb_rate=0.03, perturb_radius=10; no PNG/GIF saved for speed.

## 6. Results
Latest summary (from analysis_outputs/sweep_summary_by_mode.csv, seeds 0–9):

| boundary_mode | observer_mode         | mean_identity_score_mean | final_identity_score_mean | mean_action_rate_mean | mean_recovery_mean |
|---------------|-----------------------|--------------------------|---------------------------|-----------------------|--------------------|
| wrap          | active_persist        | 0.691                    | 0.811                     | 0.050                 | 112.687            |
| wrap          | passive_persist       | 0.679                    | 0.525                     | 0.000                 | 16.279             |
| wrap          | object_off_tracked    | 0.611                    | 0.528                     | 0.000                 | -596.244           |
| wrap          | passive_nopersist     | 0.000                    | 0.000                     | 0.000                 | -17.946            |
| zero          | active_persist        | 0.704                    | 0.739                     | 0.050                 | -70.057            |
| zero          | passive_persist       | 0.692                    | 0.733                     | 0.000                 | 35.883             |
| zero          | object_off_tracked    | 0.647                    | 0.431                     | 0.000                 | 384.954            |
| zero          | passive_nopersist     | 0.000                    | 0.000                     | 0.000                 | 6.267              |

## 7. Discussion
Active observers consistently show the highest identity continuity (identity_score) with small action budgets. Passive persistence helps but is below active; without persistence, identity collapses. Wrap vs zero backgrounds preserve the ordering; wrap shifts recovery and identity modestly.

## 8. Limitations
Binary states only; hand-set thresholds and budgets; identity is spatial/continuity-based, not semantic; shocks are simple flips.

## 9. Future Work
- Learnable or adaptive budgets.
- Richer state/action spaces.
- Longer-range or multi-entity tracking and competition.

## Appendix A: Repro commands
From repo root:
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_sweep.ps1 -seeds 0,1,2 -steps 300
.venv\Scripts\python analysis\aggregate_sweep.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

