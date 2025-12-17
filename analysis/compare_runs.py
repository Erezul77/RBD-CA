import argparse
import csv
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path: str) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in series:
                    series[k] = []
                if k == "tag":
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


def name_from_series(series: Dict[str, List[float]], path: str) -> str:
    if "tag" in series and len(series["tag"]) > 0 and isinstance(series["tag"][0], str):
        tag = series["tag"][0].strip()
        if tag:
            return tag
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem


def main():
    ap = argparse.ArgumentParser(description="Compare identity persistence metrics across runs.")
    ap.add_argument("runs", nargs="+", help="Paths to metrics.csv files.")
    ap.add_argument("--out", default="analysis_outputs", help="Directory to write comparison plots.")
    ap.add_argument("--summary-csv", default=None, help="If set, write aggregated summary by run name/tag.")
    ap.add_argument("--glob", action="store_true", help="Treat provided paths as globs.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    input_paths = []
    if args.glob:
        import glob
        for pattern in args.runs:
            input_paths.extend(glob.glob(pattern))
    else:
        input_paths = args.runs

    metrics_to_plot = ["core_size", "core_jaccard_prev", "identity_score", "identity_break", "identity_run_len", "core_size_delta", "observer_action_rate"]
    table_rows = []
    warnings: List[str] = []

    for path in input_paths:
        series = read_metrics(path)
        name = name_from_series(series, path)

        def safe_last(key: str) -> float:
            vals = series.get(key, [])
            return vals[-1] if vals else float("nan")

        def safe_sum(key: str) -> float:
            vals = series.get(key, [])
            a = np.array(vals, dtype=float)
            return float(np.nansum(a)) if a.size else float("nan")

        row = {
            "name": name,
            "mean_core_size": nanmean(series.get("core_size", [])),
            "mean_persist_iou": nanmean(series.get("persist_iou", [])),
            "mean_core_jaccard_prev": nanmean(series.get("core_jaccard_prev", [])),
            "mean_core_size_delta": nanmean(series.get("core_size_delta", [])),
            "mean_identity_run_len": nanmean(series.get("identity_run_len", [])),
            "mean_identity_score": nanmean(series.get("identity_score", [])),
            "mean_observer_action_rate": nanmean(series.get("observer_action_rate", [])),
            "final_core_size": safe_last("core_size"),
            "final_persist_iou": safe_last("persist_iou"),
            "final_core_jaccard_prev": safe_last("core_jaccard_prev"),
            "final_core_size_delta": safe_last("core_size_delta"),
            "final_identity_score": safe_last("identity_score"),
            "total_identity_breaks": safe_sum("identity_break"),
        }
        table_rows.append(row)

    # Print compact comparison table
    headers = [
        "name",
        "mean_core_size",
        "mean_persist_iou",
        "mean_core_jaccard_prev",
        "mean_core_size_delta",
        "mean_identity_run_len",
        "mean_identity_score",
        "mean_observer_action_rate",
        "final_core_size",
        "final_persist_iou",
        "final_core_jaccard_prev",
        "final_core_size_delta",
        "final_identity_score",
        "total_identity_breaks",
    ]
    print("Comparison:")
    print("\t".join(headers))
    for r in table_rows:
        print("\t".join(f"{r[h]:.4g}" if isinstance(r[h], float) else str(r[h]) for h in headers))

    # Plots
    colors = ["#4444aa", "#aa4444", "#44aa44"]
    for metric in metrics_to_plot:
        plt.figure()
        plotted = False
        for idx, path in enumerate(input_paths):
            series = read_metrics(path)
            name = name_from_series(series, path)
            if metric not in series:
                warnings.append(f"Missing metric '{metric}' in {path}, skipped plot trace.")
                continue
            t = series.get("t", list(range(len(series.get(metric, [])))))
            y = series.get(metric, [])
            plt.plot(t, y, label=name, color=colors[idx % len(colors)], linewidth=1.6)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("t")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        outfile = os.path.join(args.out, f"{metric}.png")
        plt.savefig(outfile, dpi=150)
        plt.close()

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")

    if args.summary_csv:
        agg: Dict[str, Dict[str, List[float]]] = {}
        numeric_fields = [h for h in headers if h != "name"]
        for row in table_rows:
            name = row["name"]
            if name not in agg:
                agg[name] = {f: [] for f in numeric_fields}
            for f in numeric_fields:
                agg[name][f].append(float(row[f]))
        with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name"] + numeric_fields)
            for name, data in agg.items():
                means = [np.nanmean(data[f]) if len(data[f]) else float("nan") for f in numeric_fields]
                writer.writerow([name] + means)
        print(f"Summary written to {args.summary_csv}")


if __name__ == "__main__":
    main()

