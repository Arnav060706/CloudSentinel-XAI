"""
Phase 4 regression test: the LIVE ParallelMLEngine feeds XGBoost's 1 - P(Normal)
into anomaly_score (the signal the risk engine consumes) and preserves the
Isolation Forest score as if_anomaly_score, with a fallback to the IF score when
XGBoost's attack probability is unavailable.

Guards against train/serve skew: it re-derives 1 - P(Normal) from the SAME
single-event encoding the engine used and asserts equality, so the live path and
the offline eval compute the identical per-event score.

Skipped automatically if the trained model bundles aren't present.

Run:  PYTHONPATH=. pytest -q tests/test_live_signal_wiring.py
"""
import asyncio
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.services.ml_inference import ParallelMLEngine
from app.parser_normalizer.src.pipeline import ParserPipeline

ROOT = Path(__file__).resolve().parent.parent
ISO = ROOT / "models" / "iso_forest.pkl"
XGB = ROOT / "models" / "xgboost_classifier.pkl"

pytestmark = pytest.mark.skipif(
    not (ISO.exists() and XGB.exists()),
    reason="trained model bundles not present (run models/train_*.py first)",
)


def _bruteforce_event():
    raw = {
        "userIdentity": {"type": "IAMUser", "principalId": "AIDAATTACKER",
                         "arn": "arn:aws:iam::112233445566:user/mallory.k",
                         "accountId": "112233445566", "userName": "mallory.k"},
        "eventTime": "2026-06-04T02:00:00Z", "eventSource": "signin.amazonaws.com",
        "eventName": "ConsoleLogin", "awsRegion": "us-east-1",
        "sourceIPAddress": "185.220.101.47", "userAgent": "curl/7.88.1",
        "requestParameters": None, "responseElements": {"ConsoleLogin": "Failure"},
        "additionalEventData": {"MFAUsed": "No"}, "eventID": "evt-test-1",
        "eventType": "AwsConsoleSignIn", "readOnly": False, "managementEvent": True,
        "errorCode": "FailedAuthentication", "errorMessage": "FailedAuthentication",
        "ml_labels": {"anomaly_flag": True, "threat_category": "BruteForce",
                      "severity_score": 0.8},
    }
    return ParserPipeline().process_log(raw, source_cloud="AWS").model_dump(exclude={"raw_log"})


@pytest.fixture(scope="module")
def engine():
    state = {"iso_forest_bundle": pickle.load(open(ISO, "rb")),
             "xgboost_bundle": pickle.load(open(XGB, "rb"))}
    return ParallelMLEngine(state)


def test_anomaly_score_is_one_minus_p_normal(engine):
    norm = _bruteforce_event()
    out = asyncio.run(engine.execute_parallel_inference(dict(norm)))

    # Re-derive on the same single-event encoding (train/serve parity check).
    x_iso, x_xgb, _ = engine._prepare_feature_vectors(dict(norm))
    proba = np.asarray(engine.xgboost_model.predict_proba(x_xgb)[0])
    expected_attack = round(float(1.0 - proba[engine.class_names.index("Normal")]), 4)
    iso_expected = engine._run_isolation_forest(x_iso)

    assert "if_anomaly_score" in out
    assert out["anomaly_score"] == pytest.approx(expected_attack, abs=1e-6)
    assert out["if_anomaly_score"] == pytest.approx(iso_expected, abs=1e-6)
    # This is a confident brute-force: XGBoost's signal is high and distinct
    # from the IF's, which is the whole point of wiring it in.
    assert out["predicted_phase"] == "BruteForce"
    assert out["anomaly_score"] > 0.9
    assert out["anomaly_score"] != out["if_anomaly_score"]


def test_falls_back_to_if_score_without_normal_class(engine):
    norm = _bruteforce_event()
    engine.xgb_normal_index = None  # simulate a bundle with no "Normal" class
    try:
        out = asyncio.run(engine.execute_parallel_inference(dict(norm)))
        assert out["anomaly_score"] == out["if_anomaly_score"]
    finally:
        # restore so fixture reuse (module scope) isn't corrupted for other tests
        engine.xgb_normal_index = engine.class_names.index("Normal")
