"""
Phase 2 tests: label-swept threshold selection (scoring_utils.sweep_threshold)
and the RiskEngine p99 -> p95 recalibration default.

Run:  PYTHONPATH=. pytest -q tests/test_threshold_calibration.py
"""
import numpy as np
import pytest

from app.services.risk_engine import RiskEngine
from models.scoring_utils import sweep_threshold


def test_sweep_threshold_separates_a_clean_signal():
    # Benign scores low, attacks high, with a clean gap around 0.5.
    rng = np.random.default_rng(0)
    benign = rng.uniform(0.0, 0.4, size=500)
    attack = rng.uniform(0.6, 1.0, size=50)
    scores = np.concatenate([benign, attack])
    y = np.concatenate([np.zeros(500), np.ones(50)]).astype(int)

    out = sweep_threshold(scores, y, metric="f1")
    # A clean gap -> the max-F1 cut lands in/at it (between max benign and the
    # lowest attack score inclusive) and recovers the attacks.
    assert 0.4 <= out["threshold"] <= 0.65
    assert out["f1"] > 0.95
    assert out["pr_auc"] > 0.95


def test_sweep_threshold_precision_floor():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.uniform(0, 0.5, 400), rng.uniform(0.5, 1.0, 40)])
    y = np.concatenate([np.zeros(400), np.ones(40)]).astype(int)
    out = sweep_threshold(scores, y, metric="precision_floor", precision_floor=0.9)
    assert out["precision"] >= 0.9


def test_sweep_threshold_requires_both_classes():
    with pytest.raises(ValueError):
        sweep_threshold([0.1, 0.2, 0.3], [0, 0, 0])


def test_recalibration_default_is_p95_and_configurable():
    # New default percentile is 95, not the old fragile 99.
    assert RiskEngine().recalibration_percentile == 95.0

    vals = list(range(100))  # risk_intensity 0..99
    p95 = RiskEngine().recalibrate_threshold(vals)          # uses instance default (95)
    p99 = RiskEngine().recalibrate_threshold(vals, percentile=99.0)
    assert p95 < p99  # lower percentile -> lower (less fragile) threshold

    # Constructor override is honored when no explicit percentile is passed.
    eng = RiskEngine(recalibration_percentile=90.0)
    assert eng.recalibrate_threshold(vals) == eng.recalibrate_threshold(vals, percentile=90.0)


def test_recalibration_percentile_validated():
    with pytest.raises(ValueError):
        RiskEngine(recalibration_percentile=0)
    with pytest.raises(ValueError):
        RiskEngine(recalibration_percentile=150)
