# Observer Identity Lock Experiments (Plain English)

## Definitions
- **Grid / Agents**: Binary cells on a 2D lattice; state=1 is “active,” state=0 is background medium.
- **Boundary / Skin**: Where active cells meet medium (boundary-mode=zero) or meet another pattern (boundary-mode=wrap).
- **Entity / Core**: A tracked region (largest or persistence-selected component) the system treats as “the same” across time.
- **Identity Lock (Conatus)**: Pressure to keep tracking the same core; measured by identity_score, core_jaccard_prev, run lengths between identity breaks.
- **Observer**:
  - Passive: perceives/segments and tracks identity; no causal intervention.
  - Active: tracks and can intervene locally (budgeted overrides) to stabilize identity.
- **Boundary Modes**:
  - zero: background medium with fixed state 0 outside; technical condition, not “vacuum.”
  - wrap: toroidal world; no outside; boundaries are relational.

## Model (DBC-CA)
- Moore-neighborhood binary CA with distance-to-boundary thresholds a(d), b(d).
- Boundary computed per boundary-mode (zero: padded; wrap: toroidal).
- Optional observer window (Chebyshev radius), attention drift toward core centroid.

## Observer Modes
- off_tracked: no influence; can log global core if enabled.
- passive_nopersist / passive_persist: segmentation + tracking; persistence bias optional; no rule modification.
- active_persist: segmentation + tracking + budgeted intervention inside window (difference mask vs base update, capped by observer_budget).

## Metrics
- identity_score (priority: persist_iou else core_jaccard_prev)
- core_jaccard_prev, identity_break, identity_run_len
- core_size, core_size_delta (recovery post-shock)
- observer_action_rate (active only)
- boundary_mode, observer_mode tags

## Experiments
- Sweep over boundary-mode {zero, wrap} and observer modes (off_tracked, passive_nopersist, passive_persist, active_persist) across seeds.
- Perturbation shock at t=150; recovery measured via core_size_delta.
- Metrics-only runs (no frames) for speed.

## Findings (template)
- Active observer generally improves identity_score with bounded action_rate.
- Compare zero vs wrap: wrap removes “outside,” boundaries become purely relational; check whether identity continuity shifts.

## Limitations
- Binary CA only; no multi-state agents.
- Identity is tied to spatial continuity; semantic identity not modeled.
- Budgeting and window size are hand-set; not learned.

