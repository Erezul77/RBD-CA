import argparse
import csv
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def parse_run_dir_name(run_dir_name: str):
    """
    Expected patterns:
      object_off_tracked_zero_seed0
      passive_persist_wrap_seed2
    Returns: (mode, boundary_mode, seed)
    """
    parts = run_dir_name.split("_")

    seed = None
    boundary_mode = None
    mode = run_dir_name

    if parts and re.fullmatch(r"seed\d+", parts[-1]):
        seed = int(parts[-1].replace("seed", ""))

    if len(parts) >= 2 and parts[-2] in ("zero", "wrap"):
        boundary_mode = parts[-2]
        mode = "_".join(parts[:-2])
    else:
        if seed is not None:
            mode = "_".join(parts[:-1])

    return mode, boundary_mode, seed


def read_metrics(path: str) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in series:
                    series[k] = []
                if k in ("boundary_mode", "observer_mode", "tag"):
                    series[k].append(v)
                    continue
                try:
                    series[k].append(float(v))
                except ValueError:
                    series[k].append(float("nan"))
    return series


def nanmean(arr: List[float]) -> float:
    a = np.array(arr, dtype=float)
    return float(np.nanmean(a)) if a.size else float("nan")


def nanstd(arr: List[float]) -> float:
    a = np.array(arr, dtype=float)
    return float(np.nanstd(a)) if a.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Aggregate sweep results.")
    ap.add_argument("--manifest", default="analysis_outputs/sweep_manifest.csv", help="CSV manifest listing metrics paths.")
    ap.add_argument("--out", default="analysis_outputs", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    runs = []
    manifest = Path("analysis_outputs") / "sweep_manifest.txt"
    metrics_paths = []
    if manifest.exists():
        lines = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        metrics_paths.extend(lines)
    if not metrics_paths:
        metrics_paths = [str(p) for p in Path("outputs").glob("*/metrics.csv")]

    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            header = f.readline()
            f.seek(0)
            if header.startswith("tag,") or header.startswith("tag;") or header.lower().startswith("tag"):
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_paths.append(row.get("metrics_path", "").strip())

    metrics_paths = [p for p in metrics_paths if p]

    for mp in metrics_paths:
        metrics_path = Path(mp)
        run_dir = metrics_path.parent
        mode, boundary_mode, seed = parse_run_dir_name(run_dir.name)
        runs.append({
            "tag": run_dir.name,
            "boundary_mode": boundary_mode or "unknown",
            "observer_mode": mode,
            "mode": mode,
            "seed": seed if seed is not None else -1,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
        })

    per_run = []

    for row in runs:
        path = row["metrics_path"]
        if not os.path.isfile(path):
            continue
        series = read_metrics(path)
        t = series.get("t", [])
        core_delta = series.get("core_size_delta", [])
        mask_recovery = [i for i, tv in enumerate(t) if 150 <= tv <= 250]
        recovery_vals = [core_delta[i] for i in mask_recovery] if core_delta else []
        entry = {
            "tag": row.get("tag", ""),
            "boundary_mode": row.get("boundary_mode", "unknown"),
            "observer_mode": row.get("observer_mode", "unknown"),
            "mode": row.get("mode", row.get("observer_mode", "unknown")),
            "seed": row.get("seed", -1),
            "final_identity_score": series.get("identity_score", [float("nan")])[-1] if series.get("identity_score") else float("nan"),
            "mean_identity_score": nanmean(series.get("identity_score", [])),
            "final_core_size_delta": series.get("core_size_delta", [float("nan")])[-1] if series.get("core_size_delta") else float("nan"),
            "mean_recovery_150_250": nanmean(recovery_vals),
            "mean_observer_action_rate": nanmean(series.get("observer_action_rate", [])),
        }
        per_run.append(entry)

    # Write per-seed summary
    per_seed_csv = os.path.join(args.out, "sweep_summary_by_seed.csv")
    seed_fields = list(per_run[0].keys()) if per_run else []
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(per_run)

    # Aggregate by boundary_mode + observer_mode
    agg = defaultdict(lambda: defaultdict(list))
    for r in per_run:
        key = (r["boundary_mode"], r["observer_mode"])
        for k in ("mean_identity_score", "final_identity_score", "mean_recovery_150_250", "mean_observer_action_rate"):
            agg[key][k].append(r[k])

    agg_rows = []
    for (b, o), vals in agg.items():
        agg_rows.append({
            "boundary_mode": b,
            "observer_mode": o,
            "mean_identity_score_mean": nanmean(vals["mean_identity_score"]),
            "mean_identity_score_std": nanstd(vals["mean_identity_score"]),
            "final_identity_score_mean": nanmean(vals["final_identity_score"]),
            "final_identity_score_std": nanstd(vals["final_identity_score"]),
            "mean_recovery_mean": nanmean(vals["mean_recovery_150_250"]),
            "mean_recovery_std": nanstd(vals["mean_recovery_150_250"]),
            "mean_action_rate_mean": nanmean(vals["mean_observer_action_rate"]),
            "mean_action_rate_std": nanstd(vals["mean_observer_action_rate"]),
        })

    by_mode_csv = os.path.join(args.out, "sweep_summary_by_mode.csv")
    if agg_rows:
        mode_fields = list(agg_rows[0].keys())
        with open(by_mode_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=mode_fields)
            writer.writeheader()
            writer.writerows(agg_rows)

    # Plots
    # identity by mode
    if agg_rows:
        labels = [f"{r['boundary_mode']}-{r['observer_mode']}" for r in agg_rows]
        means = [r["mean_identity_score_mean"] for r in agg_rows]
        stds = [r["mean_identity_score_std"] for r in agg_rows]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, means, yerr=stds, capsize=4)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean identity_score")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig_identity_by_mode.png"), dpi=150)
        plt.close()

        # recovery
        rec_means = [r["mean_recovery_mean"] for r in agg_rows]
        rec_stds = [r["mean_recovery_std"] for r in agg_rows]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, rec_means, yerr=rec_stds, capsize=4, color="#88aadd")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean recovery (core_size_delta 150-250)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig_recovery_by_mode.png"), dpi=150)
        plt.close()

        # action rate
        act_means = [r["mean_action_rate_mean"] for r in agg_rows]
        act_stds = [r["mean_action_rate_std"] for r in agg_rows]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, act_means, yerr=act_stds, capsize=4, color="#dd8888")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean observer_action_rate")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "fig_action_rate_by_mode.png"), dpi=150)
        plt.close()

    # Markdown report
    report_path = os.path.join(args.out, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sweep Report\n\n")
        f.write(f"Manifest: {args.manifest}\n\n")
        f.write("## Runs\n")
        for r in per_run:
            f.write(f"- {r['tag']} (boundary={r['boundary_mode']}, observer={r['observer_mode']}, seed={r['seed']})\n")
        f.write("\n## Per-mode summary (mean ± std)\n")
        for r in agg_rows:
            f.write(f"- {r['boundary_mode']}/{r['observer_mode']}: identity_score mean={r['mean_identity_score_mean']:.3f}±{r['mean_identity_score_std']:.3f}, recovery mean={r['mean_recovery_mean']:.3f}±{r['mean_recovery_std']:.3f}, action_rate mean={r['mean_action_rate_mean']:.3f}±{r['mean_action_rate_std']:.3f}\n")
        f.write("\n## Figures\n")
        f.write("- fig_identity_by_mode.png\n")
        f.write("- fig_recovery_by_mode.png\n")
        f.write("- fig_action_rate_by_mode.png\n")

    print(f"Wrote summaries to {by_mode_csv} and {per_seed_csv}")
    print(f"Report: {report_path}")

    # Optional: refresh paper table if build_report.py exists
    build_report = Path("paper") / "build_report.py"
    if build_report.exists():
        try:
            subprocess.run(
                [os.sys.executable, str(build_report)],
                check=True,
                cwd=Path(".").resolve(),
            )
        except Exception as exc:
            print(f"Warning: build_report.py failed: {exc}")


if __name__ == "__main__":
    main()

