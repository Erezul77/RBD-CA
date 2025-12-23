"""
Regenerate all tables and figures from sweep data.
Ensures consistency across all outputs.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_all_metrics(output_dir: str) -> list[dict]:
    """Load all metrics.csv files from output directory."""
    all_rows = []
    
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        csv_path = os.path.join(folder_path, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: missing metrics.csv in {folder}")
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


def parse_experiment_mode(tag: str) -> str:
    """Extract experiment mode from tag (e.g., 'passive_persist_zero_seed0' -> 'passive_persist')"""
    parts = tag.split("_")
    if len(parts) >= 2:
        if parts[0] == "object":
            return "object_off_tracked"
        elif parts[0] == "passive":
            if parts[1] in ["persist", "nopersist"]:
                return f"passive_{parts[1]}"
        elif parts[0] == "active":
            if parts[1] in ["persist", "random"]:
                return f"active_{parts[1]}"
    return tag


def aggregate_by_run(rows: list[dict]) -> dict:
    """Aggregate per-timestep data into per-run means."""
    runs = defaultdict(list)
    
    for row in rows:
        try:
            seed = int(row.get("seed", -1))
            boundary_mode = row.get("boundary_mode", "unknown")
            tag = row.get("tag", "")
            experiment_mode = parse_experiment_mode(tag)
            
            S_coh = float(row.get("S_coh", 0))
            S_pred = float(row.get("S_pred", 0))
            S_total = float(row.get("S_total", 0))
            identity_score = float(row.get("identity_score", -1))
            observer_action_rate = float(row.get("observer_action_rate", 0))
            
            key = (seed, boundary_mode, experiment_mode)
            runs[key].append({
                "S_coh": S_coh,
                "S_pred": S_pred,
                "S_total": S_total,
                "identity_score": identity_score,
                "observer_action_rate": observer_action_rate,
            })
        except (ValueError, KeyError):
            continue
    
    run_means = {}
    for key, timesteps in runs.items():
        if len(timesteps) == 0:
            continue
        
        valid_identity = [t["identity_score"] for t in timesteps if t["identity_score"] >= 0]
        
        run_means[key] = {
            "S_coh_mean": np.mean([t["S_coh"] for t in timesteps]),
            "S_pred_mean": np.mean([t["S_pred"] for t in timesteps]),
            "S_total_mean": np.mean([t["S_total"] for t in timesteps]),
            "identity_mean": np.mean(valid_identity) if valid_identity else np.nan,
            "A_hat_mean": np.mean([np.exp(-t["S_total"]) for t in timesteps]),
            "action_rate_mean": np.mean([t["observer_action_rate"] for t in timesteps]),
        }
    
    return run_means


def compute_mode_statistics(run_means: dict) -> dict:
    """Compute mean ± SD for each (boundary, mode) condition."""
    conditions = defaultdict(list)
    
    for (seed, boundary_mode, experiment_mode), stats in run_means.items():
        key = (boundary_mode, experiment_mode)
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
            "identity_mean": np.nanmean([s["identity_mean"] for s in stats_list]),
            "identity_std": np.nanstd([s["identity_mean"] for s in stats_list]),
            "A_hat_mean": np.mean([s["A_hat_mean"] for s in stats_list]),
            "A_hat_std": np.std([s["A_hat_mean"] for s in stats_list]),
            "action_rate_mean": np.mean([s["action_rate_mean"] for s in stats_list]),
            "action_rate_std": np.std([s["action_rate_mean"] for s in stats_list]),
            "n": len(stats_list),
        }
    
    return results


def compute_correlations(run_means: dict) -> dict:
    """Compute Pearson correlations for Figure 5."""
    results = {}
    
    for boundary in ["zero", "wrap"]:
        all_A = []
        all_identity = []
        persist_A = []
        persist_identity = []
        
        for (seed, bmode, omode), stats in run_means.items():
            if bmode != boundary:
                continue
            
            A_hat = stats["A_hat_mean"]
            identity = stats["identity_mean"]
            
            if np.isnan(A_hat) or np.isnan(identity):
                continue
            
            all_A.append(A_hat)
            all_identity.append(identity)
            
            if omode in ["passive_persist", "active_persist"]:
                persist_A.append(A_hat)
                persist_identity.append(identity)
        
        r_all = float(np.corrcoef(all_A, all_identity)[0, 1]) if len(all_A) >= 2 else None
        r_persist = float(np.corrcoef(persist_A, persist_identity)[0, 1]) if len(persist_A) >= 2 else None
        
        results[boundary] = {
            "r_all_modes": r_all,
            "n_all_modes": len(all_A),
            "r_persistence_only": r_persist,
            "n_persistence_only": len(persist_A),
        }
    
    return results


def generate_figure_4(mode_stats: dict, output_path: str):
    """Generate Figure 4: Misfit by mode."""
    boundaries = ["zero", "wrap"]
    modes_order = ["object_off_tracked", "passive_nopersist", "passive_persist", "active_persist", "active_random"]
    mode_labels = ["off\n(tracked)", "passive\n(no persist)", "passive\n(persist)", "active\n(persist)", "active\n(random)"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    bar_width = 0.25
    metrics = ["S_coh", "S_pred", "S_total"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    for ax_idx, bmode in enumerate(boundaries):
        ax = axes[ax_idx]
        x = np.arange(len(modes_order))
        
        for i, metric in enumerate(metrics):
            means = []
            stds = []
            for mode in modes_order:
                key = (bmode, mode)
                if key in mode_stats:
                    means.append(mode_stats[key][f"{metric}_mean"])
                    stds.append(mode_stats[key][f"{metric}_std"])
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - 1) * bar_width
            label = rf"$S_{{{metric.split('_')[1]}}}$" if metric != "S_total" else r"$S_{total}$"
            ax.bar(x + offset, means, bar_width, yerr=stds, label=label, color=colors[i], alpha=0.8, capsize=3)
        
        ax.set_xlabel("Observer Mode", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, fontsize=9)
        ax.set_title(f"Boundary: {bmode}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel("Misfit Cost", fontsize=11)
            ax.legend(loc="upper right")
    
    fig.suptitle("Figure 4: Coherence and Predictive Costs by Observer Mode", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_figure_5(run_means: dict, correlations: dict, output_path: str):
    """Generate Figure 5: A_hat vs identity."""
    boundaries = ["zero", "wrap"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    mode_colors = {
        "object_off_tracked": "#95a5a6",
        "passive_nopersist": "#e67e22",
        "passive_persist": "#3498db",
        "active_persist": "#e74c3c",
        "active_random": "#9b59b6",
    }
    
    for ax_idx, bmode in enumerate(boundaries):
        ax = axes[ax_idx]
        
        mode_data = defaultdict(lambda: {"A": [], "identity": []})
        
        for (seed, boundary_mode, omode), stats in run_means.items():
            if boundary_mode != bmode:
                continue
            
            A_hat = stats["A_hat_mean"]
            identity = stats["identity_mean"]
            
            if np.isnan(A_hat) or np.isnan(identity):
                continue
            
            mode_data[omode]["A"].append(A_hat)
            mode_data[omode]["identity"].append(identity)
        
        for mode, data in mode_data.items():
            color = mode_colors.get(mode, "#34495e")
            label = mode.replace("_", " ").replace("off tracked", "off (tracked)")
            ax.scatter(data["A"], data["identity"], alpha=0.6, s=40, label=label, c=color)
        
        # Annotate correlation
        corr = correlations[bmode]
        ax.annotate(f"Pearson r = {corr['r_all_modes']:.3f}\nn = {corr['n_all_modes']}\n\n"
                   f"Persist-only r = {corr['r_persistence_only']:.3f}\nn = {corr['n_persistence_only']}", 
                   xy=(0.05, 0.95), xycoords="axes fraction",
                   fontsize=10, fontweight="bold", verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        ax.set_xlabel(r"Adequacy $\hat{A} = \exp(-S_{total})$", fontsize=11)
        ax.set_ylabel("Identity Score", fontsize=11)
        ax.set_title(f"Boundary: {bmode}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    
    fig.suptitle(r"Figure 5: TRA Bridge Validation — $\hat{A}$ vs Identity Score", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_table_2_latex(mode_stats: dict, n_seeds: int) -> str:
    """Generate LaTeX for Table 2."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Explicit misfit metrics across $n={n_seeds}$ seeds. Mean $\pm$ SD per condition.}}",
        r"\label{tab:misfit}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Boundary & Observer & $S_{\mathrm{coh}}$ & $S_{\mathrm{pred}}$ & $S_{\mathrm{total}}$ \\",
        r"\midrule",
    ]
    
    modes_order = ["object_off_tracked", "passive_nopersist", "passive_persist", "active_persist", "active_random"]
    mode_labels = {
        "object_off_tracked": "off (tracked)",
        "passive_nopersist": "passive (no persist)",
        "passive_persist": "passive (persist)",
        "active_persist": "active (persist)",
        "active_random": "active\\_random",
    }
    
    for bmode in ["zero", "wrap"]:
        for mode in modes_order:
            key = (bmode, mode)
            if key not in mode_stats:
                continue
            s = mode_stats[key]
            label = mode_labels.get(mode, mode)
            
            # Bold the lowest S_total for each boundary
            s_total_str = f"{s['S_total_mean']:.3f} $\\pm$ {s['S_total_std']:.3f}"
            if mode == "active_persist":
                s_total_str = r"\textbf{" + s_total_str + "}"
            
            lines.append(f"{bmode} & {label} & {s['S_coh_mean']:.3f} $\\pm$ {s['S_coh_std']:.3f} & "
                        f"{s['S_pred_mean']:.3f} $\\pm$ {s['S_pred_std']:.3f} & {s_total_str} \\\\")
        
        if bmode == "zero":
            lines.append(r"\midrule")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    output_dir = "outputs_sweep50"
    figs_dir = "paper/figs"
    
    os.makedirs(figs_dir, exist_ok=True)
    
    print("Loading all metrics...")
    rows = load_all_metrics(output_dir)
    print(f"Loaded {len(rows)} timestep records")
    
    print("\nAggregating per-run means...")
    run_means = aggregate_by_run(rows)
    print(f"Computed means for {len(run_means)} runs")
    
    # Count unique seeds
    seeds = set(k[0] for k in run_means.keys())
    n_seeds = len(seeds)
    print(f"Unique seeds: {n_seeds} (range: {min(seeds)}-{max(seeds)})")
    
    print("\nComputing mode statistics...")
    mode_stats = compute_mode_statistics(run_means)
    
    print("\nComputing correlations...")
    correlations = compute_correlations(run_means)
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    print(f"\nTotal runs: {len(run_means)}")
    print(f"Expected: 5 modes × 2 boundaries × {n_seeds} seeds = {5*2*n_seeds}")
    
    print("\n=== Figure 5 Correlations ===")
    print(json.dumps(correlations, indent=2))
    
    print("\n=== Mode Statistics ===")
    for key in sorted(mode_stats.keys()):
        s = mode_stats[key]
        print(f"{key[0]}/{key[1]}: identity={s['identity_mean']:.4f}±{s['identity_std']:.4f}, "
              f"S_total={s['S_total_mean']:.3f}±{s['S_total_std']:.3f}, n={s['n']}")
    
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)
    
    print("\nGenerating Figure 4...")
    generate_figure_4(mode_stats, os.path.join(figs_dir, "fig4_misfit_by_mode.png"))
    
    print("\nGenerating Figure 5...")
    generate_figure_5(run_means, correlations, os.path.join(figs_dir, "fig5_adeq_vs_identity.png"))
    
    print("\n=== Table 2 LaTeX ===")
    table2_latex = generate_table_2_latex(mode_stats, n_seeds)
    print(table2_latex)
    
    # Save summary JSON
    summary = {
        "n_runs": len(run_means),
        "n_seeds": n_seeds,
        "seed_range": [min(seeds), max(seeds)],
        "correlations": correlations,
        "mode_statistics": {f"{k[0]}_{k[1]}": v for k, v in mode_stats.items()},
    }
    
    with open("paper/data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: paper/data_summary.json")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

