"""
Minimal end-to-end + unit tests. Runs in ML BYPASS mode (no trained models
required), so it validates that the whole pipeline is wired correctly even
before the dataset/models exist.

Run:  PYTHONPATH=. pytest -q
"""
import json
import time
import sqlite3
import os

from fastapi.testclient import TestClient


def _raw_logs():
    here = os.path.join(
        os.path.dirname(__file__), "..",
        "app", "parser_normalizer", "mock_data", "unified_datastream.json",
    )
    return json.load(open(here))


def test_health_and_ingest_end_to_end(tmp_path, monkeypatch):
    # Isolate the DB per test run
    monkeypatch.chdir(tmp_path)
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ONLINE"

        r = client.post("/api/v1/ingest/raw", json=_raw_logs())
        assert r.status_code == 202
        body = r.json()
        assert body["records_normalized"] == 3
        assert body["records_failed"] == 0

        # Allow one flush cycle (ticker = 3s)
        time.sleep(4.0)

    db = tmp_path / "cloud_sentinel.db"
    assert db.exists()
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"entity_risk_states", "xai_alerts"} <= tables
    # The 3 cross-cloud events should stitch and persist at least one entity
    assert con.execute("SELECT COUNT(*) FROM entity_risk_states").fetchone()[0] >= 1
    con.close()


def test_lifetime_cross_cloud_span():
    """Graph engine tracks a lifetime cloud footprint across paced events."""
    from app.services.graph_engine import MultiCloudGraphEngine

    g = MultiCloudGraphEngine()
    ev = {
        "principal": "arn:aws:iam::1:user/alice",
        "ua_family": "Boto3", "ua_version": "1.2",
        "is_known_proxy_or_tor": False, "principal_type": "IAMUser",
        "source_ip": "203.0.113.45", "source_cloud": "AWS",
        "timestamp": "2026-06-10T08:00:00Z",
    }
    _, _, _, _, clouds1 = g.process_event(dict(ev))
    ev2 = dict(ev, source_cloud="GCP", timestamp="2026-06-10T08:02:00Z")
    _, _, _, _, clouds2 = g.process_event(ev2)
    # Same principal -> same entity -> lifetime footprint accumulates
    assert set(clouds2) == {"AWS", "GCP"}


def test_risk_engine_uses_lifetime_clouds():
    """Cross-cloud multiplier is driven by lifetime_clouds, not the window."""
    from app.services.risk_engine import RiskEngine

    eng = RiskEngine()
    now = time.time()
    events = [{
        "principal": "p", "source_cloud": "AWS",
        "anomaly_score": 0.9,
        "timestamp": __import__("datetime").datetime.fromtimestamp(
            now, __import__("datetime").timezone.utc).isoformat(),
    }]
    single = eng.calculate_intensity(events, lifetime_clouds=["AWS"], eval_time=now)
    multi = eng.calculate_intensity(events, lifetime_clouds=["AWS", "GCP", "AZURE"], eval_time=now)
    assert multi["cloud_span_count"] == 3
    assert multi["diversity_multiplier"] > single["diversity_multiplier"]
    assert multi["risk_intensity"] > single["risk_intensity"]


def test_risk_engine_baseline_suppresses_legitimate_multicloud():
    """An identity whose baseline IS multi-cloud is not amplified (DevOps fix)."""
    from app.services.risk_engine import RiskEngine

    eng = RiskEngine()
    now = time.time()
    ts = __import__("datetime").datetime.fromtimestamp(
        now, __import__("datetime").timezone.utc).isoformat()
    events = [{"principal": "devops", "source_cloud": "AWS",
               "anomaly_score": 0.9, "timestamp": ts}]
    # Without baseline: 3 clouds amplifies.
    hot = eng.calculate_intensity(events, lifetime_clouds=["AWS", "GCP", "AZURE"],
                                  eval_time=now, entity_id="d1")
    # With baseline = those 3 clouds: no novelty, no amplification.
    eng.record_baseline("d2", ["AWS", "GCP", "AZURE"])
    calm = eng.calculate_intensity(events, lifetime_clouds=["AWS", "GCP", "AZURE"],
                                   eval_time=now, entity_id="d2")
    assert hot["diversity_multiplier"] > 1.0
    assert calm["diversity_multiplier"] == 1.0
    assert calm["novel_cloud_span_count"] == 0


def test_ml_bypass_sets_all_fields():
    """Bypass mode must return every field downstream stages read."""
    import asyncio
    from app.services.ml_inference import ParallelMLEngine

    eng = ParallelMLEngine({})  # no models -> bypass
    out = asyncio.run(eng.execute_parallel_inference({"principal": "p"}))
    for key in ("anomaly_score", "predicted_phase", "predicted_phase_index",
                "phase_confidence", "shap_attributions"):
        assert key in out