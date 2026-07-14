"""
Phase 6 regression test: xai_engine's SHAP faithfulness deletion test, rebuilt
to operate in ENGINEERED feature space with background-value deletion + a paired
random-feature control.

Exactly the README's known-good example: a ConsoleLogin failure -> BruteForce
(98.9%), top SHAP = login_result_success. Deleting the genuinely-attributed
feature(s) must collapse the BruteForce confidence MORE than deleting random
non-attributed features (gate passes); pointing the gate at a low-attribution
feature must NOT (gate fails). Also checks the gate fails CLOSED when its inputs
are missing.

Skipped automatically if the trained model bundles aren't present.

Run:  PYTHONPATH=. pytest -q tests/test_xai_faithfulness_gate.py
"""
import asyncio
import pickle
from pathlib import Path

import pytest

from app.services.ml_inference import ParallelMLEngine
from app.services.xai_engine import FaithfulnessGatedXAI
from app.parser_normalizer.src.pipeline import ParserPipeline

ROOT = Path(__file__).resolve().parent.parent
ISO = ROOT / "models" / "iso_forest.pkl"
XGB = ROOT / "models" / "xgboost_classifier.pkl"

pytestmark = pytest.mark.skipif(
    not (ISO.exists() and XGB.exists()),
    reason="trained model bundles not present (run models/train_*.py first)",
)


def _state():
    return {"iso_forest_bundle": pickle.load(open(ISO, "rb")),
            "xgboost_bundle": pickle.load(open(XGB, "rb"))}


def _scored_bruteforce_event(engine):
    raw = {
        "userIdentity": {"type": "IAMUser", "principalId": "AIDAX",
                         "arn": "arn:aws:iam::112233445566:user/mallory.k",
                         "accountId": "112233445566", "userName": "mallory.k"},
        "eventTime": "2026-06-04T02:00:00Z", "eventSource": "signin.amazonaws.com",
        "eventName": "ConsoleLogin", "awsRegion": "us-east-1",
        "sourceIPAddress": "185.220.101.47", "userAgent": "curl/7.88.1",
        "requestParameters": None, "responseElements": {"ConsoleLogin": "Failure"},
        "additionalEventData": {"MFAUsed": "No"}, "eventID": "e1",
        "eventType": "AwsConsoleSignIn", "readOnly": False, "managementEvent": True,
        "errorCode": "FailedAuthentication", "errorMessage": "FailedAuthentication",
        "ml_labels": {"anomaly_flag": True, "threat_category": "BruteForce",
                      "severity_score": 0.8},
    }
    norm = ParserPipeline().process_log(raw, source_cloud="AWS").model_dump(exclude={"raw_log"})
    return asyncio.run(engine.execute_parallel_inference(dict(norm)))


@pytest.fixture(scope="module")
def scored_and_gate():
    state = _state()
    engine = ParallelMLEngine(state, stateful_features=False)
    xai = FaithfulnessGatedXAI(state)
    scored = _scored_bruteforce_event(engine)
    return scored, xai


def test_bundle_has_background_sample():
    bundle = pickle.load(open(XGB, "rb"))
    assert "background_sample" in bundle, "run: python models/train_xgboost.py --background-only"


def test_gate_passes_for_real_attribution(scored_and_gate):
    scored, xai = scored_and_gate
    assert scored["predicted_phase"] == "BruteForce"
    assert "login_result_success" in scored["shap_attributions"]
    assert "xgb_feature_row" in scored
    passed = xai._run_deletion_test(
        scored, scored["shap_attributions"],
        scored["phase_confidence"], scored["predicted_phase_index"])
    assert passed is True


def test_gate_fails_for_low_attribution_features(scored_and_gate):
    scored, xai = scored_and_gate
    # Temporal / MFA features are not the driver of this brute-force prediction;
    # deleting them must NOT beat the random-feature control.
    for feat in ("is_weekend", "day_of_week", "hour_of_day"):
        if feat in scored["xgb_feature_row"]:
            faithful = xai._run_deletion_test(
                scored, {feat: {"shap_impact": 0.0}},
                scored["phase_confidence"], scored["predicted_phase_index"])
            assert faithful is False, f"gate wrongly passed on low-attribution {feat}"


def test_gate_fails_closed_without_feature_row(scored_and_gate):
    scored, xai = scored_and_gate
    stripped = {k: v for k, v in scored.items() if k != "xgb_feature_row"}
    assert xai._run_deletion_test(
        stripped, scored["shap_attributions"],
        scored["phase_confidence"], scored["predicted_phase_index"]) is False
