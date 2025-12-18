# Recursive Boundary Dynamics in Cellular Systems: A Boundary-Driven Cellular Automaton with Observer-Dependent Identity Stabilization

## Abstract
We present a boundary-driven cellular automaton (DBC-CA) that couples local updates to distance-from-boundary and introduces a finite observer subsystem. Boundaries are operational: active cells touching the background medium (zero) or, in wrap mode, other patterns in a toroidal world. The observer runs in three modes: off (object-view), passive (tracks a core with persistence bias but does not alter rules), and active (tracks plus budgeted local modulation of thresholds inside its window). Identity is measured by continuity of the tracked core (identity_score) and recovery after a localized shock. A sweep over boundary modes (zero, wrap) and observer modes (object_off_tracked, passive_nopersist, passive_persist, active_persist) shows that persistence bias sharply improves identity continuity, and active observers achieve the highest identity_score with a small but nonzero intervention rate. Wrap vs zero preserves the ordering, indicating the effect is not an artifact of an “outside” vacuum.

## 1. Introduction
Classic cellular automata explain emergence from neighbor rules alone. Here we invert the perspective: boundaries shape dynamics and an embedded observer can stabilize an entity’s identity. We ask whether a situated “subject-view” (finite window, memory, budgeted action) measurably increases identity persistence compared to an object-view baseline. The claim is operational, not metaphysical: observation is a causal subsystem inside the model.

## 2. Model
### 2.1 Grid and dynamics
- Binary CA on an H×W grid, Moore neighborhood, steps=300 per sweep run.
- Base rule: survival/birth thresholds a(d), b(d) depend on normalized distance to boundary (distance-to-boundary coupling).

### 2.2 Boundary
- Boundary = active cells with at least one inactive neighbor. Roughness is captured by boundary_to_active.

### 2.3 Boundary modes
- zero: fixed zero background medium; boundaries separate activity from medium.
- wrap: toroidal; no “outside,” boundaries are between patterns.

### 2.4 Observer/subject subsystem
- Attention window: Chebyshev radius (default 18); attention drifts toward core at bounded speed.
- Core selection: largest interior component in window; with persistence bias, prefers overlap with a halo of the previous core.
- Modes:
  - off_tracked: no window/influence; optional global core tracking when requested.
  - passive_nopersist: window + core selection, but no persistence bias.
  - passive_persist: window + persistence bias; no rule modulation.
  - active_persist: same as passive_persist plus local widening of thresholds inside window, capped by observer_budget (fraction of window cells allowed to override per step).
- “Observation is action”: only in active mode; observer_field modulates thresholds locally, and only a budgeted subset of differing cells is applied.

### 2.5 Perturbation
- One-time localized flip shock at t=150 with radius 10 and rate 0.03 around the attention center (when enabled).

## 3. Metrics
All metrics are written per step to metrics.csv.
- identity_score: persist_iou when available; otherwise core_jaccard_prev; else -1.
- persist_iou: hit-rate of current core against a dilated halo of the previous core.
- core_jaccard_prev: Jaccard overlap of current vs previous core.
- core_centroid_drift: distance between attention center and core centroid.
- boundary_to_active: boundary size / active size.
- core_size_delta: core_size minus baseline at perturb_t (post-shock recovery); NaN before/without shock.
- observer_action_rate: applied overrides / window size (active mode only).
- identity_break / identity_run_len: identity breaks when core_jaccard_prev < threshold; run length counts steps since last break.
- boundary_mode, observer_mode, tag recorded for aggregation.

## 4. Experiments (sweep design)
- Script: `scripts/run_sweep.ps1`.
- Modes: object_off_tracked, passive_nopersist, passive_persist, active_persist.
- Boundary modes: zero and wrap.
- Seeds: 0–29 (run in batches to avoid timeouts).
- Steps: 300; perturbation on with t=150, rate=0.03, radius=10.
- Outputs: metrics.csv per run; manifest at analysis_outputs/sweep_manifest.txt; aggregation writes summaries and plots to analysis_outputs/.

## 5. Results
Summary by (boundary_mode, observer_mode) from `analysis_outputs/sweep_summary_by_mode.csv` / `paper/results_table.md` (seeds 0–29, means ± std):

| boundary_mode | observer_mode | mean_identity_score_mean | final_identity_score_mean | mean_action_rate_mean | mean_recovery_mean |
| --- | --- | --- | --- | --- | --- |
| wrap | active_persist | 0.699 | 0.686 | 0.050 | 43.888 |
| wrap | passive_persist | 0.674 | 0.642 | 0.000 | -32.649 |
| wrap | object_off_tracked | 0.609 | 0.656 | 0.000 | -499.693 |
| wrap | passive_nopersist | 0.000 | 0.000 | 0.000 | 11.810 |
| zero | active_persist | 0.699 | 0.790 | 0.050 | -52.358 |
| zero | passive_persist | 0.681 | 0.812 | 0.000 | -2.167 |
| zero | object_off_tracked | 0.644 | 0.581 | 0.000 | 395.023 |
| zero | passive_nopersist | 0.000 | 0.000 | 0.000 | 8.767 |

Figures (copied to `paper/figures/`):
- `fig_identity_by_mode.png`
- `fig_recovery_by_mode.png`
- `fig_action_rate_by_mode.png`

Interpretation (data-backed):
- Persistence matters: passive_persist outperforms passive_nopersist in both boundary modes.
- Active_persist is highest on identity_score with a small intervention rate (~0.05 of window).
- Ordering is stable across zero vs wrap, suggesting the effect is not tied to a vacuum boundary.
- Recovery (core_size_delta) varies by mode and boundary; action is budgeted and limited.

Discussion takeaways:
- Persistence is necessary: nopersist modes collapse identity (~0).
- Active observer improves identity with a small action budget.
- Wrap vs zero preserves ranking, so effects are not artifacts of an “outside vacuum.”

## 6. Discussion (operational ontology)
- Entity: a boundary-maintained pattern whose core is tracked over time.
- Identity lock: halo-based persistence bias that maintains continuity; identity_break flags when continuity drops.
- Observer: a causal subsystem internal to the CA; passive = selection/tracking only; active = selection plus bounded intervention.
- This is not a claim about consciousness; it is a minimal operational model where subject-view changes identity statistics.

## 7. Limitations
- Finite seeds (0–29 in current manifest); stochastic sensitivity remains.
- Heuristic observer and hand-tuned thresholds/budgets.
- Binary, 2D medium; identity is spatial/overlap-based.
- Recovery metric tied to a single shock schedule.

## 8. Future Work
- Extend sweeps to seeds 0–49 for tighter confidence.
- Competing observers and multi-entity tracking.
- Learned or adaptive observers (optimize budgets/fields).
- Multi-state or higher-dimensional media; richer perturbations.

## 9. Conclusion
Boundary-driven CA supports entities whose identity can be stabilized by a subject-like subsystem. Persistence bias alone boosts identity continuity; bounded active intervention improves it further, and the effect holds across zero and wrap media.

