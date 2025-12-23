"""
Verify and recompute statistics for reproducibility check.
"""
import os
import json
import numpy as np
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
    """Aggregate per-timestep data into per-run means."""
    runs = defaultdict(list)
    
    for row in rows:
        try:
            seed = int(row.get("seed", -1))
            boundary_mode = row.get("boundary_mode", "unknown")
            # Use tag to get full experiment mode (e.g., passive_persist vs passive_nopersist)
            tag = row.get("tag", "")
            
            # Extract experiment mode from tag (e.g., "passive_persist_zero_seed0" -> "passive_persist")
            parts = tag.split("_")
            if len(parts) >= 2:
                if parts[0] in ["object", "passive", "active"]:
                    if parts[0] == "object":
                        experiment_mode = "object_off_tracked"
                    elif parts[0] == "passive":
                        experiment_mode = f"passive_{parts[1]}" if parts[1] in ["persist", "nopersist"] else "passive"
                    elif parts[0] == "active":
                        experiment_mode = f"active_{parts[1]}" if parts[1] in ["persist", "random"] else "active"
                    else:
                        experiment_mode = parts[0]
                else:
                    experiment_mode = row.get("observer_mode", "unknown")
            else:
                experiment_mode = row.get("observer_mode", "unknown")
            
            S_total = float(row.get("S_total", 0))
            identity_score = float(row.get("identity_score", -1))
            
            key = (seed, boundary_mode, experiment_mode)
            runs[key].append({
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
        
        valid_identity = [t["identity_score"] for t in timesteps if t["identity_score"] >= 0]
        
        run_means[key] = {
            "S_total_mean": np.mean([t["S_total"] for t in timesteps]),
            "identity_score_mean": np.mean(valid_identity) if valid_identity else np.nan,
            "A_hat_mean": np.mean([np.exp(-t["S_total"]) for t in timesteps]),
        }
    
    return run_means


def compute_correlations(run_means: dict) -> dict:
    """Compute Pearson correlations for Figure 5."""
    results = {}
    
    for boundary in ["zero", "wrap"]:
        # All modes
        all_A = []
        all_identity = []
        
        # Persistence modes only (passive_persist + active_persist)
        persist_A = []
        persist_identity = []
        
        for (seed, bmode, omode), stats in run_means.items():
            if bmode != boundary:
                continue
            
            A_hat = stats["A_hat_mean"]
            identity = stats["identity_score_mean"]
            
            if np.isnan(A_hat) or np.isnan(identity):
                continue
            
            all_A.append(A_hat)
            all_identity.append(identity)
            
            # Check if persistence mode (passive_persist or active_persist)
            if omode in ["passive_persist", "active_persist"]:
                persist_A.append(A_hat)
                persist_identity.append(identity)
        
        # Compute correlations
        if len(all_A) >= 2:
            r_all = float(np.corrcoef(all_A, all_identity)[0, 1])
        else:
            r_all = None
        
        if len(persist_A) >= 2:
            r_persist = float(np.corrcoef(persist_A, persist_identity)[0, 1])
        else:
            r_persist = None
        
        results[boundary] = {
            "r_all_modes": r_all,
            "n_all_modes": len(all_A),
            "r_persistence_only": r_persist,
            "n_persistence_only": len(persist_A),
        }
    
    return results


def compute_mode_statistics(run_means: dict) -> dict:
    """Compute mean ± SD for each condition."""
    from collections import defaultdict
    
    conditions = defaultdict(list)
    
    for (seed, boundary_mode, observer_mode), stats in run_means.items():
        key = (boundary_mode, observer_mode)
        conditions[key].append(stats)
    
    results = {}
    for key, stats_list in conditions.items():
        S_coh_values = []
        S_pred_values = []
        S_total_values = []
        identity_values = []
        
        for s in stats_list:
            S_total_values.append(s["S_total_mean"])
            identity_values.append(s["identity_score_mean"])
        
        results[f"{key[0]}_{key[1]}"] = {
            "S_total_mean": float(np.mean(S_total_values)),
            "S_total_std": float(np.std(S_total_values)),
            "identity_mean": float(np.nanmean(identity_values)),
            "identity_std": float(np.nanstd(identity_values)),
            "n": len(stats_list),
        }
    
    return results


def main():
    output_dir = "outputs_sweep50"
    
    print("Loading all metrics...")
    rows = load_all_metrics(output_dir)
    print(f"Loaded {len(rows)} timestep records")
    
    print("\nAggregating per-run means...")
    run_means = aggregate_by_run(rows)
    print(f"Computed means for {len(run_means)} runs")
    
    print("\n=== Figure 5 Correlations (recomputed from CSVs) ===")
    correlations = compute_correlations(run_means)
    print(json.dumps(correlations, indent=2))
    
    print("\n=== Mode Statistics ===")
    mode_stats = compute_mode_statistics(run_means)
    for key in sorted(mode_stats.keys()):
        stats = mode_stats[key]
        print(f"{key}: S_total={stats['S_total_mean']:.3f}±{stats['S_total_std']:.3f}, "
              f"identity={stats['identity_mean']:.4f}±{stats['identity_std']:.4f}, n={stats['n']}")


if __name__ == "__main__":
    main()

