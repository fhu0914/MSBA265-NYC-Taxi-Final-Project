"""MLOps step 1 of 2: build the monitoring reference baseline.

Records what "normal" looked like for the model at training time, so that
src/monitor.py can later measure how far live traffic has moved away from it.

Run from the project root AFTER src/p2_analysis.py:
    python src/make_monitoring_baseline.py

Data: data/train.zip, downloaded separately from Kaggle (see README.md).
Output: artifacts/monitoring_baseline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "train.zip"
ARTIFACT = ROOT / "artifacts" / "taxi_eta_pipeline.joblib"
BASELINE = ROOT / "artifacts" / "monitoring_baseline.json"
SEED = 42

# Number of quantile bins used to describe each distribution. Ten bins is the
# usual choice for PSI: enough resolution to see a shift, few enough that each
# bin still holds a meaningful number of rows.
N_BINS = 10


def haversine_km(df: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"].to_numpy() - df["pickup_longitude"].to_numpy())
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predict_minutes(model, x: pd.DataFrame) -> np.ndarray:
    log_seconds = model.predict(x)
    return np.clip(np.maximum(np.expm1(np.clip(log_seconds, -10, 30)), 0) / 60, 0, 360)


def reference_bins(values: np.ndarray) -> dict:
    """Describe one variable as quantile bin edges plus the share of rows per bin.

    Quantile edges (not equal-width edges) are used so that every reference bin
    starts out with roughly the same share of the training rows. That keeps the
    PSI denominator away from zero and makes the index stable.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
    # Open the outer edges so that live values beyond the training range still
    # land in the first or last bin instead of being dropped. The open ends are
    # stored as null, because JSON has no literal for infinity; src/monitor.py
    # turns them back into -inf and +inf when it loads the file.
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(values, bins=edges)[0]
    return {
        "edges": [None if not np.isfinite(e) else float(e) for e in edges],
        "proportions": [float(p) for p in counts / counts.sum()],
        "mean": float(np.mean(values)),
        "p50": float(np.median(values)),
    }


if not DATA.exists():
    raise SystemExit(f"Missing {DATA}. Read README.md to download Kaggle train.zip (do not unzip).")
if not ARTIFACT.exists():
    raise SystemExit(f"Missing {ARTIFACT}. Run python src/p2_analysis.py first.")

train = pd.read_csv(
    DATA,
    usecols=["pickup_datetime", "pickup_latitude", "pickup_longitude",
             "dropoff_latitude", "dropoff_longitude", "trip_duration"],
)
train = train.dropna().copy()
train = train[train.trip_duration > 0].copy()
train["pickup_hour"] = pd.to_datetime(train.pickup_datetime, errors="coerce").dt.hour
train = train.dropna(subset=["pickup_hour"])
train["pickup_hour"] = train["pickup_hour"].astype(int)
train["distance_km"] = haversine_km(train)
train = train[np.isfinite(train.distance_km)].copy()
# Same scope rule as src/p2_analysis.py, so the baseline describes exactly the
# population the deployed model was fitted on.
train = train[(train.distance_km <= 200) & (train.trip_duration <= 6 * 3600)].copy()

# Same split and seed as src/p2_analysis.py: the baseline must come from the
# training rows only, never from validation rows.
train_set, _ = train_test_split(train, test_size=.20, random_state=SEED)

bundle = joblib.load(ARTIFACT)
features = bundle["features"]
train_pred = predict_minutes(bundle["model"], train_set[features])

baseline = {
    "created_from": "training split of data/train.zip (scoped, seed 42)",
    "training_rows": int(len(train_set)),
    "n_bins": N_BINS,
    "psi_thresholds": {"stable": 0.10, "investigate": 0.25},
    "features": {
        "distance_km": reference_bins(train_set.distance_km.to_numpy()),
        "pickup_hour": reference_bins(train_set.pickup_hour.to_numpy().astype(float)),
    },
    "prediction": {"predicted_minutes": reference_bins(train_pred)},
}

BASELINE.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
print(f"Baseline written to {BASELINE.relative_to(ROOT)}")
print(f"Reference rows: {len(train_set):,}; bins per variable: {N_BINS}")
for name, ref in baseline["features"].items():
    print(f"  {name}: mean {ref['mean']:.3f}, median {ref['p50']:.3f}")
print(f"  predicted_minutes: mean {baseline['prediction']['predicted_minutes']['mean']:.3f}")
