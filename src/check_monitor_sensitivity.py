"""Reproducible evidence for the MIN_REQUESTS setting in src/monitor.py.

A drift alarm is only useful if it stays quiet when nothing is wrong. This
script measures the false-alarm rate directly: it repeatedly draws traffic that
matches the training baseline by construction, runs the same PSI function the
monitor uses, and counts how often each verdict fires anyway. Any firing is a
false alarm, because the traffic came from the reference distribution.

Run from the project root, after src/make_monitoring_baseline.py:
    python src/check_monitor_sensitivity.py

Output: reports/monitor_sensitivity.csv and a printed table.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Running this as "python src/check_monitor_sensitivity.py" puts src/ on the
# import path, not the project root, so the package import below would fail.
sys.path.insert(0, str(ROOT))

from src.monitor import classify, population_stability_index  # noqa: E402

BASELINE = ROOT / "artifacts" / "monitoring_baseline.json"
OUT = ROOT / "reports" / "monitor_sensitivity.csv"

SAMPLE_SIZES = [30, 60, 100, 200, 500, 1000, 2000]
TRIALS = 400
SEED = 0


def finite_edges(reference: dict) -> np.ndarray:
    """Close the two open outer edges so values can be drawn inside them."""
    edges = list(reference["edges"])
    edges[0] = edges[1] - (edges[2] - edges[1])
    edges[-1] = edges[-2] + (edges[-2] - edges[-3])
    return np.array(edges, dtype=float)


def matching_sample(reference: dict, edges: np.ndarray, rng, size: int) -> np.ndarray:
    """Traffic drawn from the reference distribution itself: no real drift."""
    probs = np.array(reference["proportions"], dtype=float)
    bins = rng.choice(len(probs), size=size, p=probs / probs.sum())
    return rng.uniform(edges[:-1][bins], edges[1:][bins])


def main() -> None:
    if not BASELINE.exists():
        raise SystemExit(
            f"Missing {BASELINE}. Run python src/make_monitoring_baseline.py first."
        )
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    stable = baseline["psi_thresholds"]["stable"]
    investigate = baseline["psi_thresholds"]["investigate"]
    reference = baseline["features"]["distance_km"]
    edges = finite_edges(reference)
    rng = np.random.default_rng(SEED)

    rows = []
    for n in SAMPLE_SIZES:
        psis = np.array([
            population_stability_index(reference, matching_sample(reference, edges, rng, n))
            for _ in range(TRIALS)
        ])
        verdicts = [classify(p, stable, investigate) for p in psis]
        rows.append({
            "requests": n,
            "median_psi": round(float(np.median(psis)), 4),
            "p95_psi": round(float(np.quantile(psis, 0.95)), 4),
            "false_watch_pct": round(100 * float(np.mean([v != "stable" for v in verdicts])), 1),
            "false_investigate_pct": round(100 * float(np.mean([v == "investigate" for v in verdicts])), 1),
        })

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Baseline: {baseline['training_rows']:,} training rows, "
          f"{baseline['n_bins']} bins; {TRIALS} trials per size, seed {SEED}.")
    print("Traffic matches the baseline by construction, so every alarm below is false.\n")
    print(f"{'requests':>9} {'median PSI':>11} {'p95 PSI':>9} {'false watch':>12} {'false investigate':>18}")
    for r in rows:
        print(f"{r['requests']:>9} {r['median_psi']:>11.4f} {r['p95_psi']:>9.4f} "
              f"{r['false_watch_pct']:>11.1f}% {r['false_investigate_pct']:>17.1f}%")
    print(f"\nWritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
