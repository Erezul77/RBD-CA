import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


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
    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)

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
            "tag": row["tag"],
            "boundary_mode": row["boundary_mode"],
            "observer_mode": row["observer_mode"],
            "seed": row["seed"],
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


if __name__ == "__main__":
    main()

