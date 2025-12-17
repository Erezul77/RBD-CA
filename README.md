# RBD-CA

This project explores boundary-driven cellular automata (DBC-CA) with a “subject-view” observer.

## Concepts
- **Object-view**: plain CA update from neighbor sums and distance-to-boundary thresholds. No attention or persistence; structures drift and mix.
- **Subject-view**: an attention window (Chebyshev radius) with a chosen **core** (largest or best-continuing component) inside it. The subject locally widens survival thresholds near the core, moves attention toward it, and keeps the window inside bounds.
- **Boundary (“skin”)**: active cells that touch at least one inactive neighbor (with fixed-zero outside). High boundary/active means rough, fragmented shapes; lower means smoother blobs.
- **Core**: candidate entity tracked inside the subject window (interior preferred). Size and overlap with previous core indicate persistence.
- **Identity lock / persistence bias (conatus)**: the subject tries to keep the same core over time (halo-based continuity). Metrics like `persist_iou`, `core_jaccard_prev`, and attention/core alignment reflect stability.

## Metrics (per step)
- `core_size`: number of cells in the current core.
- `persist_iou`: halo-hit rate for continuing the previous core (higher = better identity continuity).
- `core_jaccard_prev`: Jaccard overlap of current vs previous core (NaN if no previous core).
- `core_centroid_drift`: distance from attention center to core centroid (alignment).
- `boundary_to_active`: roughness ratio; high values mean lots of boundary relative to active mass.
- `core_size_delta`: change from pre-shock core size (after perturbation).
- `shock`: 1 on the perturbation step, else 0.
- `tag`: experiment label.
- `identity_break`: 1 when core continuity drops below a threshold.
- `identity_run_len`: steps since last identity break.

## Running
Use the virtual environment: `.venv\Scripts\python bda.py --help` for all flags.
Outputs are written under `outputs/` with PNG frames, a GIF, and `metrics.csv`.

## Operational Ontology
- **Entity**: whatever the process can keep tracking as “the same” even under perturbation (core continuity + location).
- **Identity lock (Conatus)**: persistence bias plus optional active stabilization (halo-based continuity and, if active, local threshold widening).
- **Object-view (observer off)**: patterns emerge but no privileged identity; metrics can still track the largest global component if requested. Boundaries meet a background medium (state 0), not a metaphysical vacuum.
- **Subject-view (observer passive/active)**: a finite window + memory picks and follows a core, optionally influencing local survival rules (active) to stabilize that identity.
- **Boundary dynamics (skin)**: boundary cells separate “inside” from “outside”; roughness reflects fragmentation.
- **Observer passive**: segmentation + identity lock only (epistemic action); no feedback into CA rules.
- **Observer active**: segmentation + identity lock + local feedback (dynamic action) to hold the same core.
- **Conatus pressure**: the system pushes to maximize continuity of one core over time; identity breaks reset that run length.
- **Observation as Selection (passive) vs Intervention (active)**: passive observes and maintains identity through tracking; active applies a budgeted intervention inside the attention window to keep the same core alive.

## No vacuum interpretation
- `boundary-mode=zero`: uses state 0 as a background medium (technical boundary condition), not a claim of empty space.
- `boundary-mode=wrap`: toroidal world; there is no “outside”, so boundaries are only between patterns within the closed grid.

## Operational Ontology (in this repo)
- **Boundary / skin**: where active meets background medium (zero) or meets another pattern (wrap).
- **Entity**: a tracked, identity-locked region (core) that persists over time.
- **Identity lock**: continuity pressure measured by identity_score / core_jaccard_prev; broken events reset run length.
- **Observer**: passive = segmentation and tracking; active = segmentation + tracking + budgeted local intervention.
- **Medium**: zero (background medium) vs wrap (closed world).

## What counts as evidence
- A sweep over boundary-mode {zero, wrap} and observer-mode {off_tracked, passive_nopersist, passive_persist, active_persist} across multiple seeds.
- Metrics of interest: identity_score, recovery (core_size_delta post-perturb), observer_action_rate.
- Plots and summaries in `analysis_outputs/` compare modes and boundary conditions.

