import argparse
import csv
import os
import time
from dataclasses import dataclass
from collections import deque
from typing import Tuple
import math

import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from coherence_cost import CoherenceCostCalculator

# Moore neighborhood (8-connected)
N8 = [(-1, -1), (-1, 0), (-1, 1),
      (0, -1),           (0, 1),
      (1, -1),  (1, 0),  (1, 1)]


@dataclass
class Params:
    # lattice
    H: int = 80
    W: int = 120
    steps: int = 300
    seed: int = 0
    init_density: float = 0.15

    # Base DBC-CA thresholds (distance-to-boundary):
    # a(d) = floor(a0 + a1*d), b(d) = floor(b0 + b1*d), clipped to [0,8]
    a0: float = 2.0
    a1: float = 2.0
    b0: float = 3.0
    b1: float = 3.0

    # outputs
    frame_every: int = 2
    save_png: bool = True
    save_gif: bool = True
    dpi: int = 140

    # cycle detect
    detect_cycles: bool = True
    cycle_window: int = 200

    # --- Subject-view Observer (finite, situated) ---
    subject_on: bool = False
    subject_radius: int = 18         # Chebyshev radius of perception/action window
    subject_step: int = 2            # max movement per axis per step (attention drift limit)
    subject_alpha: float = 3.0       # field decay in the subject window
    subject_beta: float = 3.0        # local promotion magnitude (widen window near core)
    subject_core_min: int = 20       # minimum core size before using it; else fallback to active-in-window
    save_subject_mask: bool = False  # save subject window + core mask images
    subject_persist: bool = True          # identity lock / persistence bias
    subject_persist_dilate: int = 2       # how far "memory halo" reaches (in steps)
    subject_persist_min_hits: int = 8     # minimal contact with halo to count as "same thing"

    # perturbation (shock) test
    perturb_on: bool = False
    perturb_t: int = 150
    perturb_rate: float = 0.03
    perturb_radius: int = 10

    # experiment tag
    experiment_tag: str = ""
    observer_mode: str = "off"            # off | passive | active
    track_global_core: bool = False

    # identity break
    identity_break_thresh: float = 0.05

    # observer action budget (active only)
    observer_budget: float = 0.05

    # edge handling
    boundary_mode: str = "zero"           # zero | wrap


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def neighbor_sum(grid: np.ndarray) -> np.ndarray:
    """Sum of 8 neighbors with fixed-zero boundary conditions."""
    H, W = grid.shape
    p = np.pad(grid, 1, mode="constant", constant_values=0)
    s = np.zeros((H, W), dtype=np.int32)
    for dy, dx in N8:
        s += p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W]
    return s


def boundary_mask(grid: np.ndarray) -> np.ndarray:
    """Boundary = active cell with at least one inactive Moore neighbor (fixed-zero outside grid)."""
    H, W = grid.shape
    p = np.pad(grid, 1, mode="constant", constant_values=0)
    active = (grid == 1)
    is_b = np.zeros((H, W), dtype=bool)
    for dy, dx in N8:
        neigh = p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W]
        is_b |= (active & (neigh == 0))
    return is_b


def distance_to_sources(sources: np.ndarray) -> np.ndarray:
    """Multi-source BFS distance on 8-neighborhood from a boolean source mask."""
    H, W = sources.shape
    INF = 10**9
    dist = np.full((H, W), INF, dtype=np.int32)

    q = deque()
    ys, xs = np.where(sources)
    for y, x in zip(ys, xs):
        dist[y, x] = 0
        q.append((int(y), int(x)))

    if not q:
        return dist

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        nd = d + 1
        for dy, dx in N8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] > nd:
                dist[ny, nx] = nd
                q.append((ny, nx))

    return dist


def normalize_distance(dist: np.ndarray) -> Tuple[np.ndarray, int]:
    """Normalize finite distances to [0,1]. Returns (dnorm, dmax_raw)."""
    INF = 10**9
    finite = (dist < INF)
    if not np.any(finite):
        return np.ones(dist.shape, dtype=np.float32), 0
    dmax = int(dist[finite].max())
    dmax = max(dmax, 1)
    return (dist.astype(np.float32) / float(dmax)).astype(np.float32), dmax


def neighbor_sum(grid: np.ndarray, boundary_mode: str) -> np.ndarray:
    if boundary_mode == "wrap":
        s = np.zeros_like(grid, dtype=np.int32)
        for dy, dx in N8:
            s += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        return s
    # fixed0 (default)
    H, W = grid.shape
    p = np.pad(grid, 1, mode="constant", constant_values=0)
    s = np.zeros((H, W), dtype=np.int32)
    for dy, dx in N8:
        s += p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W]
    return s


def boundary_mask(grid: np.ndarray, boundary_mode: str) -> np.ndarray:
    H, W = grid.shape
    active = (grid == 1)
    if boundary_mode == "wrap":
        is_b = np.zeros((H, W), dtype=bool)
        for dy, dx in N8:
            neigh = np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
            is_b |= (active & (neigh == 0))
        return is_b
    # fixed0
    p = np.pad(grid, 1, mode="constant", constant_values=0)
    is_b = np.zeros((H, W), dtype=bool)
    for dy, dx in N8:
        neigh = p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W]
        is_b |= (active & (neigh == 0))
    return is_b


def thresholds_from_distance(dnorm: np.ndarray, a0: float, a1: float, b0: float, b1: float):
    a = np.floor(a0 + a1 * dnorm).astype(np.int32)
    b = np.floor(b0 + b1 * dnorm).astype(np.int32)
    a = np.clip(a, 0, 8)
    b = np.clip(b, 0, 8)
    return a, b


def chebyshev_window_mask(H: int, W: int, cy: int, cx: int, r: int) -> np.ndarray:
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    return (np.abs(y - cy) <= r) & (np.abs(x - cx) <= r)


def centroid_of_mask(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.nan, np.nan
    return float(np.mean(ys)), float(np.mean(xs))


def largest_component(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    best = []
    ys, xs = np.where(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        # BFS over 8-neighborhood
        q = [(int(sy), int(sx))]
        visited[sy, sx] = True
        comp = [(int(sy), int(sx))]
        while q:
            y, x = q.pop()
            for dy, dx in N8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))
        if len(comp) > len(best):
            best = comp

    out = np.zeros((H, W), dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def best_component_by_overlap(mask: np.ndarray, prev_core: np.ndarray, min_iou: float) -> Tuple[np.ndarray, float]:
    """
    Returns the connected component inside 'mask' that best overlaps prev_core.
    If nothing overlaps enough, returns empty mask and best_iou (so caller can fallback).
    """
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    prev_sum = int(prev_core.sum())
    best_comp = []
    best_iou = 0.0

    ys, xs = np.where(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        q = [(int(sy), int(sx))]
        visited[sy, sx] = True
        comp = [(int(sy), int(sx))]
        inter = 1 if prev_core[sy, sx] else 0

        while q:
            y, x = q.pop()
            for dy, dx in N8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))
                    if prev_core[ny, nx]:
                        inter += 1

        comp_size = len(comp)
        union = comp_size + prev_sum - inter
        iou = (inter / union) if union > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best_comp = comp

    out = np.zeros((H, W), dtype=bool)
    for y, x in best_comp:
        out[y, x] = True

    if best_iou >= float(min_iou) and int(out.sum()) > 0:
        return out, best_iou
    return np.zeros((H, W), dtype=bool), best_iou


def dilate8(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(int(steps)):
        m = out.copy()
        out |= m
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        out[1:, 1:] |= m[:-1, :-1]
        out[1:, :-1] |= m[:-1, 1:]
        out[:-1, 1:] |= m[1:, :-1]
        out[:-1, :-1] |= m[1:, 1:]
    return out


def best_component_by_hits(mask: np.ndarray, halo: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """
    Choose the connected component in mask that has the most contact ("hits") with halo.
    Returns: component_mask, hits, hit_rate (hits/component_size)
    """
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    best = []
    best_hits = 0
    best_rate = 0.0

    ys, xs = np.where(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        q = [(int(sy), int(sx))]
        visited[sy, sx] = True
        comp = [(int(sy), int(sx))]
        hits = 1 if halo[sy, sx] else 0

        while q:
            y, x = q.pop()
            for dy, dx in N8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and (not visited[ny, nx]) and mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
                    comp.append((ny, nx))
                    if halo[ny, nx]:
                        hits += 1

        comp_size = len(comp)
        rate = (hits / comp_size) if comp_size > 0 else 0.0

        if (hits > best_hits) or (hits == best_hits and rate > best_rate):
            best_hits = hits
            best_rate = rate
            best = comp

    out = np.zeros((H, W), dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out, int(best_hits), float(best_rate)


def apply_local_perturb(grid: np.ndarray, cy: int, cx: int, radius: int, rate: float, rng: np.random.Generator) -> None:
    H, W = grid.shape
    r = int(radius)
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        dy = y - cy
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            dx = x - cx
            if dx * dx + dy * dy <= r * r:
                if rng.random() < float(rate):
                    grid[y, x] = 1 - grid[y, x]  # flip state in-place


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def save_frame_png(grid: np.ndarray, outpath: str, title: str, dpi: int = 140) -> None:
    plt.figure()
    plt.imshow(grid, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def save_mask_png(mask: np.ndarray, outpath: str, title: str, dpi: int = 140) -> None:
    plt.figure()
    plt.imshow(mask.astype(np.uint8), interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()


def grid_hash(grid: np.ndarray) -> int:
    return hash(grid.tobytes())


def step(grid: np.ndarray, params: Params, att_y: int, att_x: int, prev_core_global: np.ndarray, rng: np.random.Generator):
    H, W = grid.shape
    s = neighbor_sum(grid, params.boundary_mode)

    # system boundary + DBC distance-to-boundary
    B = boundary_mask(grid, params.boundary_mode)
    distB = distance_to_sources(B)
    dB_norm, dB_max = normalize_distance(distB)
    a_base, b_base = thresholds_from_distance(dB_norm, params.a0, params.a1, params.b0, params.b1)

    a_eff = a_base.astype(np.float32)
    b_eff = b_base.astype(np.float32)
    next_state_base = ((s >= a_base) & (s <= b_base)).astype(np.uint8)

    subject_enabled = params.observer_mode in ("passive", "active", "active_random")
    track_global_only = (params.observer_mode == "off" and params.track_global_core)

    Wmask = np.zeros((H, W), dtype=bool)
    core_global = np.zeros((H, W), dtype=bool)
    dS_max = 0
    persist_iou = -1.0
    core_jaccard_prev = float("nan")
    core_centroid_drift = float("nan")
    observer_action_cells = 0
    observer_action_rate = 0.0

    if subject_enabled:
        margin = params.subject_radius
        att_y = clamp_int(att_y, margin, H - margin - 1)
        att_x = clamp_int(att_x, margin, W - margin - 1)

        Wmask = chebyshev_window_mask(H, W, att_y, att_x, params.subject_radius)
        thing_mask = (grid == 1) & Wmask
        interior_mask = (grid == 1) & (~B) & Wmask

        if params.observer_mode == "active_random":
            # no persistence bias; choose core by current interior/active only
            core_candidate = largest_component(interior_mask) if int(interior_mask.sum()) > 0 else largest_component(thing_mask)
            core_local = core_candidate
            persist_iou = -1.0
        elif params.subject_persist and int(prev_core_global.sum()) > 0:
            halo = dilate8(prev_core_global, steps=params.subject_persist_dilate)
            keep, hits, hit_rate = best_component_by_hits(thing_mask, halo)
            persist_iou = float(hit_rate)

            if hits >= params.subject_persist_min_hits:
                core_candidate = keep & (~B)
                core_local = core_candidate if int(core_candidate.sum()) >= params.subject_core_min else keep
            else:
                core_candidate = largest_component(interior_mask) if int(interior_mask.sum()) > 0 else largest_component(thing_mask)
                core_local = core_candidate
        else:
            core_candidate = largest_component(interior_mask) if int(interior_mask.sum()) > 0 else largest_component(thing_mask)
            core_local = core_candidate

        core_global = core_local.copy()

        distS = distance_to_sources(core_global)
        dS_norm, dS_max = normalize_distance(distS)
        O = np.exp(-params.subject_alpha * dS_norm).astype(np.float32) if int(core_global.sum()) > 0 else np.zeros((H, W), dtype=np.float32)

        if params.observer_mode == "active":
            a_eff[Wmask] = a_eff[Wmask] - params.subject_beta * O[Wmask]
            b_eff[Wmask] = b_eff[Wmask] + params.subject_beta * O[Wmask]

        cy, cx = centroid_of_mask(core_global)
        if not np.isnan(cy):
            target_y = int(round(cy))
            target_x = int(round(cx))
            dy = clamp_int(target_y - att_y, -params.subject_step, params.subject_step)
            dx = clamp_int(target_x - att_x, -params.subject_step, params.subject_step)
            att_y = clamp_int(att_y + dy, 0, H - 1)
            att_x = clamp_int(att_x + dx, 0, W - 1)
            att_y = clamp_int(att_y, margin, H - margin - 1)
            att_x = clamp_int(att_x, margin, W - margin - 1)
            core_centroid_drift = float(np.hypot(cy - att_y, cx - att_x))

        inter = int((core_global & prev_core_global).sum())
        union = int((core_global | prev_core_global).sum())
        if union > 0:
            core_jaccard_prev = inter / union

    elif track_global_only:
        full_mask = (grid == 1)
        if int(prev_core_global.sum()) > 0:
            halo = dilate8(prev_core_global, steps=params.subject_persist_dilate)
            best, hits, hit_rate = best_component_by_hits(full_mask, halo)
            core_global = best if int(best.sum()) > 0 else largest_component(full_mask)
            core_jaccard_prev = hit_rate
        else:
            core_global = largest_component(full_mask)
            inter = int((core_global & prev_core_global).sum())
            union = int((core_global | prev_core_global).sum())
            if union > 0:
                core_jaccard_prev = inter / union

    # finalize thresholds
    a_eff = np.clip(np.floor(a_eff), 0, 8).astype(np.int32)
    b_eff = np.clip(np.floor(b_eff), 0, 8).astype(np.int32)
    b_eff = np.maximum(b_eff, a_eff)

    if params.observer_mode == "active":
        next_state_mod = ((s >= a_eff) & (s <= b_eff)).astype(np.uint8)
        diff_mask = (next_state_mod != next_state_base) & Wmask
        diff_indices = np.flatnonzero(diff_mask)
        budget_limit = int(math.floor(params.observer_budget * max(1, int(Wmask.sum()))))
        if diff_indices.size <= budget_limit:
            new_grid = next_state_mod
            observer_action_cells = diff_indices.size
        else:
            # deterministic subset using rng (seeded from run)
            chosen = rng.choice(diff_indices, size=budget_limit, replace=False) if budget_limit > 0 else np.array([], dtype=int)
            new_grid = next_state_base.copy()
            new_grid.flat[chosen] = next_state_mod.flat[chosen]
            observer_action_cells = chosen.size
        observer_action_rate = (observer_action_cells / float(max(1, int(Wmask.sum())))) if Wmask.sum() > 0 else 0.0
    elif params.observer_mode == "active_random":
        # random intervention within window, no persistence bias or structure guidance
        new_grid = next_state_base.copy()
        window_indices = np.flatnonzero(Wmask)
        budget_limit = int(math.floor(params.observer_budget * max(1, int(Wmask.sum()))))
        if budget_limit > 0 and window_indices.size > 0:
            chosen = rng.choice(window_indices, size=min(budget_limit, window_indices.size), replace=False)
            rand_states = rng.integers(0, 2, size=chosen.size, dtype=np.uint8)
            new_grid.flat[chosen] = rand_states
            observer_action_cells = chosen.size
        else:
            observer_action_cells = 0
        observer_action_rate = (observer_action_cells / float(max(1, int(Wmask.sum())))) if Wmask.sum() > 0 else 0.0
    else:
        new_grid = next_state_base

    # metrics on new state
    Bn = boundary_mask(new_grid, params.boundary_mode)
    active = int(new_grid.sum())
    boundary = int(Bn.sum())
    interior = int(((new_grid == 1) & (~Bn)).sum())

    # subject-window diagnostics (on current grid before update is also acceptable; we use new_grid for consistency)
    window_active = int(((new_grid == 1) & Wmask).sum()) if subject_enabled else 0
    window_interior = int(((new_grid == 1) & (~Bn) & Wmask).sum()) if subject_enabled else 0
    core_size = int(core_global.sum())
    window_size = int(Wmask.sum()) if subject_enabled else 0

    metrics = {
        "active": active,
        "boundary": boundary,
        "interior": interior,
        "boundary_to_active": float(boundary / active) if active > 0 else 0.0,
        "dB_max": int(dB_max),
        "dS_max": int(dS_max),

        # subject-view metrics
        "att_y": int(att_y) if subject_enabled else -1,
        "att_x": int(att_x) if subject_enabled else -1,
        "window_size": int(window_size),
        "window_active": int(window_active),
        "window_interior": int(window_interior),
        "core_size": int(core_size),
        "persist_iou": float(persist_iou) if subject_enabled else -1.0,
        "core_jaccard_prev": float(core_jaccard_prev),
        "core_centroid_drift": float(core_centroid_drift),
        "observer_action_cells": int(observer_action_cells),
        "observer_action_rate": float(observer_action_rate),
        "identity_score": float(persist_iou if persist_iou != -1 else core_jaccard_prev if not math.isnan(core_jaccard_prev) else -1),
        "boundary_mode": params.boundary_mode,
        "observer_mode": params.observer_mode,
    }

    return new_grid, metrics, att_y, att_x, Wmask, core_global


def run(params: Params, out_dir: str, run_name: str = "run") -> None:
    ensure_dir(out_dir)

    rng = np.random.default_rng(params.seed)
    grid = (rng.random((params.H, params.W)) < params.init_density).astype(np.uint8)
    prev_core = np.zeros_like(grid, dtype=bool)
    baseline_core_size = None
    subject_enabled = params.observer_mode in ("passive", "active")
    identity_run_len = 0

    # attention starts at center (deterministic)
    att_y = params.H // 2
    att_x = params.W // 2

    frames = []
    metrics_rows = []

    seen = {}
    cycle_found = None

    # TRA misfit calculator
    coherence_calc = CoherenceCostCalculator(
        window_size=(2 * params.subject_radius, 2 * params.subject_radius),
        memory_depth=5
    )

    for t in range(params.steps + 1):
        # save frames
        if t % params.frame_every == 0:
            if params.save_png:
                png_path = os.path.join(out_dir, f"frame_{t:04d}.png")
                tag = f"{params.observer_mode.upper()}"
                title = (
                    f"{tag}-CA V1.3 | t={t} | "
                    f"a0={params.a0},a1={params.a1}, b0={params.b0},b1={params.b1} | "
                    f"rho0={params.init_density} | seed={params.seed}"
                )
                if subject_enabled:
                    title += f" | r={params.subject_radius},beta={params.subject_beta},alpha={params.subject_alpha},step={params.subject_step}"
                    title += f" | att=({att_y},{att_x})"
                save_frame_png(grid, png_path, title, dpi=params.dpi)

            if params.save_gif and imageio is not None:
                frames.append((grid * 255).astype(np.uint8))

        # cycle detection (hash on current state)
        if params.detect_cycles and cycle_found is None:
            h = grid_hash(grid)
            if h in seen:
                cycle_found = {"first_seen_t": seen[h], "t": t, "period": t - seen[h]}
            else:
                seen[h] = t
                if len(seen) > params.cycle_window:
                    oldest_key = min(seen, key=seen.get)
                    del seen[oldest_key]

        # stop after final record
        if t == params.steps:
            break

        if params.perturb_on and t == params.perturb_t:
            apply_local_perturb(grid, att_y, att_x, params.perturb_radius, params.perturb_rate, rng)

        # step
        grid, m, att_y, att_x, Wmask, core = step(grid, params, att_y, att_x, prev_core, rng)
        prev_core = core.copy()
        m["t"] = t + 1
        m["shock"] = 1 if (params.perturb_on and t == params.perturb_t) else 0
        m["tag"] = params.experiment_tag

        # TRA misfit computation
        S_coh, S_pred, S_total = coherence_calc.update(
            current_core_mask=core,
            predicted_core_mask=None,
            observer_mode=params.observer_mode,
            intervention_mask=None,
        )
        m["S_coh"] = float(S_coh)
        m["S_pred"] = float(S_pred)
        m["S_total"] = float(S_total)
        m["seed"] = params.seed

        cj = m.get("core_jaccard_prev", float("nan"))
        if math.isnan(cj):
            identity_break = 1
        else:
            identity_break = 1 if cj < params.identity_break_thresh else 0
        identity_run_len = 0 if identity_break else identity_run_len + 1
        m["identity_break"] = identity_break
        m["identity_run_len"] = identity_run_len

        if params.perturb_on and m["t"] == params.perturb_t:
            baseline_core_size = m["core_size"]

        if params.perturb_on and baseline_core_size is not None and m["t"] > params.perturb_t:
            m["core_size_delta"] = m["core_size"] - baseline_core_size
        else:
            m["core_size_delta"] = float("nan")

        metrics_rows.append(m)

        if subject_enabled and params.save_subject_mask and (t % params.frame_every == 0):
            save_mask_png(Wmask, os.path.join(out_dir, f"window_{t:04d}.png"),
                          title=f"Subject window | t={t} | att=({att_y},{att_x})", dpi=params.dpi)
            save_mask_png(core, os.path.join(out_dir, f"core_{t:04d}.png"),
                          title=f"Subject core | t={t} | core_size={int(core.sum())}", dpi=params.dpi)

    # write metrics
    csv_path = os.path.join(out_dir, "metrics.csv")
    fieldnames = [
        "seed", "t", "active", "boundary", "interior", "boundary_to_active", "dB_max", "dS_max",
        "att_y", "att_x", "window_size", "window_active", "window_interior", "core_size",
        "persist_iou", "core_jaccard_prev", "core_centroid_drift", "identity_break", "identity_run_len",
        "shock", "core_size_delta", "observer_action_cells", "observer_action_rate", "identity_score",
        "S_coh", "S_pred", "S_total",
        "boundary_mode", "observer_mode", "tag"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    # gif
    if params.save_gif and imageio is not None and len(frames) > 0:
        gif_path = os.path.join(out_dir, f"{run_name}.gif")
        imageio.mimsave(gif_path, frames, duration=0.08)

    # summary
    final = metrics_rows[-1] if metrics_rows else {}
    print("=== DBC-CA V1.3 RUN SUMMARY ===")
    print(f"Output dir: {out_dir}")
    print(f"Grid: {params.H}x{params.W} | steps: {params.steps} | seed: {params.seed}")
    print(f"Init density: {params.init_density}")
    print(f"Base thresholds: a(d)=floor({params.a0}+{params.a1}*d), b(d)=floor({params.b0}+{params.b1}*d)")
    print(f"Subject-view: {'ON' if params.subject_on else 'OFF'}")
    if params.subject_on:
        print(f"  radius={params.subject_radius} | step={params.subject_step} | beta={params.subject_beta} | alpha={params.subject_alpha} | core_min={params.subject_core_min}")
    if final:
        print(f"Final active: {final['active']} | final boundary: {final['boundary']} | final interior: {final['interior']}")
        print(f"Final boundary/active: {final['boundary_to_active']:.3f}")
        if params.subject_on:
            print(f"Final att: ({final['att_y']},{final['att_x']}) | core_size: {final['core_size']} | window_interior: {final['window_interior']}")
    if cycle_found is not None:
        print(f"Cycle detected (hash-based): first_seen_t={cycle_found['first_seen_t']} t={cycle_found['t']} period≈{cycle_found['period']}")
    else:
        print("No cycle detected (within hash window).")


def parse_args():
    p = argparse.ArgumentParser(description="DBC-CA V1.3 + Subject-view (finite observer) option.")
    p.add_argument("--demo", action="store_true", help="Run and save outputs.")

    p.add_argument("--H", type=int, default=80)
    p.add_argument("--W", type=int, default=120)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-density", type=float, default=0.15)

    p.add_argument("--a0", type=float, default=2.0)
    p.add_argument("--a1", type=float, default=2.0)
    p.add_argument("--b0", type=float, default=3.0)
    p.add_argument("--b1", type=float, default=3.0)

    p.add_argument("--frame-every", type=int, default=2)
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--dpi", type=int, default=140)

    p.add_argument("--out", type=str, default="")
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--no-cycle-detect", action="store_true")

    # subject-view args
    p.add_argument("--subject-on", action="store_true", help="Enable finite, situated observer (local perception + local action).")
    p.add_argument("--subject-radius", type=int, default=18)
    p.add_argument("--subject-step", type=int, default=2)
    p.add_argument("--subject-alpha", type=float, default=3.0)
    p.add_argument("--subject-beta", type=float, default=3.0)
    p.add_argument("--subject-core-min", type=int, default=20)
    p.add_argument("--save-subject-mask", action="store_true", help="Save window_*.png and core_*.png for debugging.")
    p.add_argument("--no-subject-persist", action="store_true")
    p.add_argument("--subject-persist-dilate", type=int, default=2)
    p.add_argument("--subject-persist-min-hits", type=int, default=8)
    p.add_argument("--observer-mode", type=str, choices=["off", "passive", "active", "active_random"], default=None, help="off: no tracking/influence; passive: track but no influence; active: track + influence; active_random: random intervention with same budget.")
    p.add_argument("--track-global-core", action="store_true", help="Track largest global component even when observer_mode=off.")
    p.add_argument("--perturb-on", action="store_true", help="Inject a local perturbation (shock).")
    p.add_argument("--perturb-t", type=int, default=150)
    p.add_argument("--perturb-rate", type=float, default=0.03)
    p.add_argument("--perturb-radius", type=int, default=10)
    p.add_argument("--experiment-tag", type=str, default="", help="Label for run folder and metrics rows.")
    p.add_argument("--identity-break-thresh", type=float, default=0.05, help="Threshold for identity break based on core_jaccard_prev.")
    p.add_argument("--observer-budget", type=float, default=0.05, help="Fraction of window cells the active observer may override per step.")
    p.add_argument("--boundary-mode", type=str, choices=["zero", "wrap"], default="zero", help="Boundary condition: zero (state 0 background) or wrap (toroidal).")

    return p.parse_args()


def main():
    args = parse_args()

    # Backward compatibility: if subject_on provided and observer_mode not set, promote to active
    if args.observer_mode is None:
        final_mode = "active" if args.subject_on else "off"
    else:
        final_mode = args.observer_mode

    boundary_mode = args.boundary_mode

    subject_enabled = final_mode in ("passive", "active", "active_random")

    params = Params(
        H=args.H, W=args.W, steps=args.steps, seed=args.seed, init_density=args.init_density,
        a0=args.a0, a1=args.a1, b0=args.b0, b1=args.b1,
        frame_every=max(1, args.frame_every),
        save_png=not args.no_png,
        save_gif=not args.no_gif,
        dpi=args.dpi,
        detect_cycles=not args.no_cycle_detect,

        subject_on=subject_enabled,
        observer_mode=final_mode,
        track_global_core=bool(args.track_global_core),
        boundary_mode=boundary_mode,
        subject_radius=max(1, args.subject_radius),
        subject_step=max(1, args.subject_step),
        subject_alpha=float(args.subject_alpha),
        subject_beta=float(args.subject_beta),
        subject_core_min=max(1, args.subject_core_min),
        save_subject_mask=bool(args.save_subject_mask),
        subject_persist=(not args.no_subject_persist),
        subject_persist_dilate=int(args.subject_persist_dilate),
        subject_persist_min_hits=int(args.subject_persist_min_hits),
        perturb_on=bool(args.perturb_on),
        perturb_t=int(args.perturb_t),
        perturb_rate=float(args.perturb_rate),
        perturb_radius=int(args.perturb_radius),
        experiment_tag=args.experiment_tag.strip(),
        identity_break_thresh=float(args.identity_break_thresh),
        observer_budget=max(0.0, float(args.observer_budget)),
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = params.experiment_tag.replace(" ", "_")
    tag_safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)
    edge_suffix = f"_{params.boundary_mode}"
    suffix = f"{edge_suffix}_{tag_safe}" if tag_safe else edge_suffix
    out_dir = args.out.strip() or os.path.join("outputs", f"dbc_subject_v1_3_{ts}{suffix}")
    ensure_dir(out_dir)

    run(params, out_dir=out_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()

