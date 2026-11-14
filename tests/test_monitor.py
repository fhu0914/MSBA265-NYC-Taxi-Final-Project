"""Run: python -m pytest -q tests/test_monitor.py

Checks that the drift detector actually reacts: it must stay quiet on traffic
drawn from the reference distribution and raise the alarm on traffic that is
clearly different. A monitor that never fires is indistinguishable from no
monitor at all, so both directions are tested.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.monitor import classify, population_stability_index

BASELINE = Path(__file__).resolve().parent.parent / "artifacts" / "monitoring_baseline.json"

pytestmark = pytest.mark.skipif(
    not BASELINE.exists(),
    reason="Run python src/make_monitoring_baseline.py first.",
)


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def reference_sample(reference, rng, size=5000):
    """Draw a sample that follows the reference distribution.

    Each bin is picked with its stored probability, then a value is drawn
    uniformly inside that bin, so the result matches the baseline by
    construction and PSI should come out near zero.
    """
    edges = list(reference["edges"])
    # Replace the open outer edges with finite ones so a value can be drawn.
    edges[0] = edges[1] - (edges[2] - edges[1])
    edges[-1] = edges[-2] + (edges[-2] - edges[-3])
    probs = np.array(reference["proportions"], dtype=float)
    bins = rng.choice(len(probs), size=size, p=probs / probs.sum())
    lo = np.array(edges[:-1], dtype=float)[bins]
    hi = np.array(edges[1:], dtype=float)[bins]
    return rng.uniform(lo, hi)


def test_matching_traffic_is_stable(baseline):
    rng = np.random.default_rng(0)
    ref = baseline["features"]["distance_km"]
    psi = population_stability_index(ref, reference_sample(ref, rng))
    assert psi < baseline["psi_thresholds"]["stable"]
    assert classify(psi, 0.10, 0.25) == "stable"


def test_shifted_traffic_is_flagged(baseline):
    """All trips suddenly far longer than anything in the training population."""
    ref = baseline["features"]["distance_km"]
    rng = np.random.default_rng(1)
    live = rng.uniform(60, 120, size=5000)
    psi = population_stability_index(ref, live)
    assert psi > baseline["psi_thresholds"]["investigate"]
    assert classify(psi, 0.10, 0.25) == "investigate"


def test_night_only_traffic_is_flagged(baseline):
    """A feed that only sends overnight pickups is a realistic ingestion fault."""
    ref = baseline["features"]["pickup_hour"]
    rng = np.random.default_rng(2)
    live = rng.integers(0, 4, size=5000).astype(float)
    psi = population_stability_index(ref, live)
    assert psi > baseline["psi_thresholds"]["investigate"]


def test_thresholds_follow_the_standard_bands():
    assert classify(0.05, 0.10, 0.25) == "stable"
    assert classify(0.18, 0.10, 0.25) == "watch"
    assert classify(0.40, 0.10, 0.25) == "investigate"
