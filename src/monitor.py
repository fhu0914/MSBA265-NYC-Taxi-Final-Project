"""MLOps step 2 of 2: check live traffic against the training baseline.

The deployed model never sees the real trip duration, so live accuracy cannot
be measured directly. What can be measured is whether the traffic arriving at
the API still looks like the data the model was fitted on, and whether the
predictions it returns still look normal. Both are standard early-warning
signals: a model does not usually fail loudly, it fails on inputs it was never
trained for.

Run from the project root:
    python src/monitor.py                      # reads logs/predictions.jsonl
    python src/monitor.py --log <path.jsonl>   # or any other request log

Output: reports/monitoring_report.json and a printed verdict.
Exit code 1 when any variable crosses the investigate threshold, so the check
can be wired into a scheduled job later.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "artifacts" / "monitoring_baseline.json"
DEFAULT_LOG = ROOT / "logs" / "predictions.jsonl"
REPORT = ROOT / "reports" / "monitoring_report.json"

# Below this many requests PSI is dominated by sampling noise rather than by any
# real shift, so the run reports "insufficient data" instead of a verdict.
#
# The number is measured, not assumed. src/check_monitor_sensitivity.py draws
# traffic that matches the baseline by construction and records how often the
# index fires anyway:
#
#     requests   false "watch"   false "investigate"
#         30         97.2%             68.2%
#         60         75.8%             14.8%
#        100         40.2%              0.2%
#        200          2.8%              0.0%
#        500          0.0%              0.0%
#
# At 30 requests a perfectly healthy service would be flagged for investigation
# two times out of three, which would train the operator to ignore the alarm.
# 500 is the first round figure where both false-alarm rates reach zero.
MIN_REQUESTS = 500


def population_stability_index(reference: dict, live: np.ndarray) -> float:
    """PSI between the stored reference distribution and a live sample.

        PSI = sum over bins of (live_share - ref_share) * ln(live_share / ref_share)

    It is zero when the two distributions match and grows as they separate.
    Empty bins are floored at a small value so the logarithm stays finite; the
    floor is small enough not to mask a real shift.
    """
    # The baseline stores the two open outer edges as null (JSON has no
    # infinity literal); restore them so out-of-range live values still fall
    # into the first or last bin rather than being discarded.
    raw = list(reference["edges"])
    raw[0] = -np.inf if raw[0] is None else raw[0]
    raw[-1] = np.inf if raw[-1] is None else raw[-1]
    edges = np.array(raw, dtype=float)
    ref_share = np.array(reference["proportions"], dtype=float)
    live_counts = np.histogram(live, bins=edges)[0]
    live_share = live_counts / max(live_counts.sum(), 1)

    floor = 1e-6
    ref_share = np.clip(ref_share, floor, None)
    live_share = np.clip(live_share, floor, None)
    return float(np.sum((live_share - ref_share) * np.log(live_share / ref_share)))


def classify(psi: float, stable: float, investigate: float) -> str:
    """Industry-standard PSI reading, used here unchanged.

    < 0.10  no meaningful shift
    0.10-0.25  moderate shift, keep watching
    > 0.25  significant shift, investigate before trusting the output
    """
    if psi < stable:
        return "stable"
    if psi < investigate:
        return "watch"
    return "investigate"


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No request log at {path}.\n"
            "Start the API (uvicorn app:app --reload) and send some /predict "
            "requests first, or pass --log with another JSONL file."
        )
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A log being appended to while this runs can end in a partial line.
            continue
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Check live API traffic for drift.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG,
                        help="JSONL request log (default: logs/predictions.jsonl)")
    args = parser.parse_args()

    if not BASELINE.exists():
        raise SystemExit(
            f"Missing {BASELINE}.\n"
            "Run python src/make_monitoring_baseline.py first."
        )

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    stable = baseline["psi_thresholds"]["stable"]
    investigate = baseline["psi_thresholds"]["investigate"]
    records = load_log(args.log)

    tracked = {
        "distance_km": baseline["features"]["distance_km"],
        "pickup_hour": baseline["features"]["pickup_hour"],
        "predicted_minutes": baseline["prediction"]["predicted_minutes"],
    }

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "log": str(args.log.relative_to(ROOT)) if args.log.is_relative_to(ROOT) else str(args.log),
        "requests": len(records),
        "min_requests": MIN_REQUESTS,
        "psi_thresholds": baseline["psi_thresholds"],
        "variables": {},
    }

    if len(records) < MIN_REQUESTS:
        report["verdict"] = "insufficient data"
        report["note"] = (
            f"Only {len(records)} requests logged; at least {MIN_REQUESTS} are needed "
            "before a PSI reading is meaningful."
        )
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Insufficient data: {len(records)} requests (need {MIN_REQUESTS}).")
        print(f"Report written to {REPORT.relative_to(ROOT)}")
        return 0

    worst = "stable"
    order = {"stable": 0, "watch": 1, "investigate": 2}
    for name, reference in tracked.items():
        values = np.array([r[name] for r in records if name in r], dtype=float)
        if values.size == 0:
            continue
        psi = population_stability_index(reference, values)
        status = classify(psi, stable, investigate)
        if order[status] > order[worst]:
            worst = status
        report["variables"][name] = {
            "psi": round(psi, 4),
            "status": status,
            "live_mean": round(float(values.mean()), 3),
            "baseline_mean": round(float(reference["mean"]), 3),
        }

    report["verdict"] = worst
    report["action"] = {
        "stable": "No action. Live traffic matches the training population.",
        "watch": "Re-check after more traffic accumulates. Do not retrain yet.",
        "investigate": (
            "Investigate before trusting the estimates: confirm the input source is "
            "unchanged, compare against recent actual trip durations, and consider "
            "refitting on data that covers the new population."
        ),
    }[worst]

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Requests checked: {len(records)}")
    for name, row in report["variables"].items():
        print(f"  {name:<18} PSI {row['psi']:>7.4f}  {row['status']:<12} "
              f"live mean {row['live_mean']:>8.3f} vs baseline {row['baseline_mean']:.3f}")
    print(f"Verdict: {worst.upper()}")
    print(report["action"])
    print(f"Report written to {REPORT.relative_to(ROOT)}")
    return 1 if worst == "investigate" else 0


if __name__ == "__main__":
    sys.exit(main())
