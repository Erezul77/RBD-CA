"""
Generate Figures 4 and 5 from the logged sweep CSV data.

Figure 4: Mean ± SD of S_coh, S_pred, S_total by observer mode
Figure 5: Scatter of A_hat = exp(-S_total) vs identity_score with Pearson r

Author: Generated for TRA Masterpaper
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_sweep_data(manifest_path: str) -> list[dict]:
    """Load all metrics.csv files listed in the manifest."""
    all_rows = []
    
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        csv_paths = [line.strip() for line in f if line.strip()]
    
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                values = line.strip().split(",")
                if len(values) != len(header):
                    continue
                row = dict(zip(header, values))
                all_rows.append(row)
    
    return all_rows


def aggregate_by_run(rows: list[dict]) -> dict:
    """
    Aggregate per-timestep data into per-run means.
    Returns dict keyed by (seed, boundary_mode, observer_mode).
    """
    runs = defaultdict(list)
    
    for row in rows:
        try:
            seed = int(row.get("seed", -1))
            boundary_mode = row.get("boundary_mode", "unknown")
            observer_mode = row.get("observer_mode", "unknown")
            
            S_coh = float(row.get("S_coh", 0))
            S_pred = float(row.get("S_pred", 0))
            S_total = float(row.get("S_total", 0))
            identity_score = float(row.get("identity_score", -1))
            
            key = (seed, boundary_mode, observer_mode)
            runs[key].append({
                "S_coh": S_coh,
                "S_pred": S_pred,
                "S_total": S_total,
                "identity_score": identity_score,
            })
        except (ValueError, KeyError):
            continue
    
    # Compute per-run means
    run_means = {}
    for key, timesteps in runs.items():
        if len(timesteps) == 0:
            continue
        
        # Filter valid identity scores
        valid_identity = [t["identity_score"] for t in timesteps if t["identity_score"] >= 0]
        
        run_means[key] = {
            "S_coh_mean": np.mean([t["S_coh"] for t in timesteps]),
            "S_pred_mean": np.mean([t["S_pred"] for t in timesteps]),
            "S_total_mean": np.mean([t["S_total"] for t in timesteps]),
            "identity_score_mean": np.mean(valid_identity) if valid_identity else np.nan,
            "A_hat_mean": np.mean([np.exp(-t["S_total"]) for t in timesteps]),
            "n_timesteps": len(timesteps),
        }
    
    return run_means


def compute_mode_statistics(run_means: dict) -> dict:
    """
    Compute mean ± SD across seeds for each (boundary_mode, observer_mode) condition.
    """
    conditions = defaultdict(list)
    
    for (seed, boundary_mode, observer_mode), stats in run_means.items():
        key = (boundary_mode, observer_mode)
        conditions[key].append(stats)
    
    results = {}
    for key, stats_list in conditions.items():
        results[key] = {
            "S_coh_mean": np.mean([s["S_coh_mean"] for s in stats_list]),
            "S_coh_std": np.std([s["S_coh_mean"] for s in stats_list]),
            "S_pred_mean": np.mean([s["S_pred_mean"] for s in stats_list]),
            "S_pred_std": np.std([s["S_pred_mean"] for s in stats_list]),
            "S_total_mean": np.mean([s["S_total_mean"] for s in stats_list]),
            "S_total_std": np.std([s["S_total_mean"] for s in stats_list]),
            "A_hat_mean": np.mean([s["A_hat_mean"] for s in stats_list]),
            "A_hat_std": np.std([s["A_hat_mean"] for s in stats_list]),
            "identity_mean": np.nanmean([s["identity_score_mean"] for s in stats_list]),
            "identity_std": np.nanstd([s["identity_score_mean"] for s in stats_list]),
            "n_seeds": len(stats_list),
        }
    
    return results


def create_figure_4(mode_stats: dict, output_path: str):
    """
    Create Figure 4: Bar plots showing mean ± SD of S_coh, S_pred, S_total by mode.
    Separate panels for each boundary mode.
    """
    boundary_modes = sorted(set(k[0] for k in mode_stats.keys()))
    observer_modes = ["object_off_tracked", "passive_nopersist", "passive_persist", "active_persist", "active_random"]
    
    # Filter to only modes that exist in data
    available_modes = set(k[1] for k in mode_stats.keys())
    # Map experiment tags to cleaner labels
    mode_labels = {
        "off": "Off (tracked)",
        "passive": "Passive",
        "active": "Active",
        "active_random": "Active Random",
    }
    
    fig, axes = plt.subplots(1, len(boundary_modes), figsize=(14, 6), sharey=True)
    if len(boundary_modes) == 1:
        axes = [axes]
    
    bar_width = 0.25
    metrics = ["S_coh", "S_pred", "S_total"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    for ax_idx, bmode in enumerate(boundary_modes):
        ax = axes[ax_idx]
        
        # Get modes for this boundary
        modes_in_data = [k[1] for k in mode_stats.keys() if k[0] == bmode]
        modes_in_data = sorted(set(modes_in_data))
        
        x = np.arange(len(modes_in_data))
        
        for i, metric in enumerate(metrics):
            means = []
            stds = []
            for mode in modes_in_data:
                key = (bmode, mode)
                if key in mode_stats:
                    means.append(mode_stats[key][f"{metric}_mean"])
                    stds.append(mode_stats[key][f"{metric}_std"])
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, means, bar_width, yerr=stds, 
                         label=rf"$S_{{{metric.split('_')[1]}}}$" if i < 2 else r"$S_{total}$",
                         color=colors[i], alpha=0.8, capsize=3)
        
        ax.set_xlabel("Observer Mode", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes_in_data], fontsize=9)
        ax.set_title(f"Boundary: {bmode}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel("Misfit Cost", fontsize=11)
            ax.legend(loc="upper right")
    
    fig.suptitle("Figure 4: Coherence and Predictive Costs by Observer Mode", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4 to: {output_path}")


def create_figure_5(run_means: dict, output_path: str):
    """
    Create Figure 5: Scatter plot of A_hat vs identity_score with Pearson r.
    """
    boundary_modes = sorted(set(k[1] for k in run_means.keys()))
    
    fig, axes = plt.subplots(1, len(boundary_modes), figsize=(14, 6))
    if len(boundary_modes) == 1:
        axes = [axes]
    
    # Color map for observer modes
    mode_colors = {
        "off": "#95a5a6",
        "passive": "#3498db",
        "active": "#e74c3c",
        "active_random": "#9b59b6",
    }
    
    for ax_idx, bmode in enumerate(boundary_modes):
        ax = axes[ax_idx]
        
        all_A = []
        all_identity = []
        
        mode_data = defaultdict(lambda: {"A": [], "identity": []})
        
        for (seed, boundary_mode, observer_mode), stats in run_means.items():
            if boundary_mode != bmode:
                continue
            
            A_hat = stats["A_hat_mean"]
            identity = stats["identity_score_mean"]
            
            if np.isnan(A_hat) or np.isnan(identity):
                continue
            
            all_A.append(A_hat)
            all_identity.append(identity)
            mode_data[observer_mode]["A"].append(A_hat)
            mode_data[observer_mode]["identity"].append(identity)
        
        # Plot by mode
        for mode, data in mode_data.items():
            color = mode_colors.get(mode, "#34495e")
            ax.scatter(data["A"], data["identity"], alpha=0.6, s=40, 
                      label=mode.replace("_", " ").title(), c=color)
        
        # Compute and annotate Pearson r
        if len(all_A) > 2:
            r = np.corrcoef(all_A, all_identity)[0, 1]
            ax.annotate(f"Pearson r = {r:.3f}\nn = {len(all_A)}", 
                       xy=(0.05, 0.95), xycoords="axes fraction",
                       fontsize=11, fontweight="bold",
                       verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        ax.set_xlabel(r"Adequacy $\hat{A} = \exp(-S_{total})$", fontsize=11)
        ax.set_ylabel("Identity Score", fontsize=11)
        ax.set_title(f"Boundary: {bmode}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    
    fig.suptitle(r"Figure 5: TRA Bridge Validation — $\hat{A}$ vs Identity Score", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 5 to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Figures 4-5 from sweep data")
    parser.add_argument("--manifest", type=str, default="analysis_outputs/sweep_manifest.txt",
                       help="Path to sweep manifest file")
    parser.add_argument("--output-dir", type=str, default="paper/figs",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from manifest: {args.manifest}")
    rows = load_sweep_data(args.manifest)
    print(f"Loaded {len(rows)} timestep records")
    
    if len(rows) == 0:
        print("ERROR: No data loaded. Check manifest path.")
        return
    
    print("Aggregating per-run means...")
    run_means = aggregate_by_run(rows)
    print(f"Computed means for {len(run_means)} runs")
    
    print("Computing mode statistics...")
    mode_stats = compute_mode_statistics(run_means)
    
    # Print summary table
    print("\n=== Mode Statistics Summary ===")
    print(f"{'Boundary':<10} {'Mode':<20} {'S_coh':<15} {'S_pred':<15} {'S_total':<15} {'n'}")
    for (bmode, omode), stats in sorted(mode_stats.items()):
        print(f"{bmode:<10} {omode:<20} {stats['S_coh_mean']:.3f}±{stats['S_coh_std']:.3f}    "
              f"{stats['S_pred_mean']:.3f}±{stats['S_pred_std']:.3f}    "
              f"{stats['S_total_mean']:.3f}±{stats['S_total_std']:.3f}    {stats['n_seeds']}")
    
    # Generate figures
    fig4_path = os.path.join(args.output_dir, "fig4_misfit_by_mode.png")
    fig5_path = os.path.join(args.output_dir, "fig5_adeq_vs_identity.png")
    
    print("\nGenerating Figure 4...")
    create_figure_4(mode_stats, fig4_path)
    
    print("Generating Figure 5...")
    create_figure_5(run_means, fig5_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

