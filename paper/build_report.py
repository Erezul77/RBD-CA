import csv
import os
import sys
from pathlib import Path


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def format_float(val, fmt="0.3f"):
    try:
        return format(float(val), fmt)
    except Exception:
        return str(val)


def main():
    # Avoid Windows console encoding errors
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    root = Path(__file__).resolve().parent.parent
    src = root / "analysis_outputs" / "sweep_summary_by_mode.csv"
    out = root / "paper" / "results_table.md"

    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    rows = load_rows(src)

    # sort by boundary_mode then mean_identity_score_mean desc
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("boundary_mode", ""), -float(r.get("mean_identity_score_mean", "nan") or 0.0))
    )

    header = [
        "boundary_mode",
        "observer_mode",
        "mean_identity_score_mean",
        "final_identity_score_mean",
        "mean_action_rate_mean",
        "mean_recovery_mean",
    ]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows_sorted:
        lines.append("| " + " | ".join([
            r.get("boundary_mode", ""),
            r.get("observer_mode", ""),
            format_float(r.get("mean_identity_score_mean", "nan")),
            format_float(r.get("final_identity_score_mean", "nan")),
            format_float(r.get("mean_action_rate_mean", "nan")),
            format_float(r.get("mean_recovery_mean", "nan")),
        ]) + " |")

    os.makedirs(out.parent, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    try:
        print(f"Wrote {out}")
    except Exception:
        # Fallback for stubborn consoles
        sys.stdout.buffer.write(f"Wrote {out}\n".encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()

