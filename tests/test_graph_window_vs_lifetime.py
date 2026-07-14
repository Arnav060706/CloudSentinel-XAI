"""
Phase 1d regression test: the cross-cloud multiplier must be driven by an
entity's LIFETIME cloud footprint, not the 60s sliding window.

Two events for the SAME principal in different clouds, 120s apart, are never
co-resident in a 60s window. If the multiplier still sees 2 clouds, it reads the
lifetime footprint (correct). If a future change ever re-couples the multiplier
to the window, a slow-paced cross-cloud campaign silently loses amplification —
this test fails loudly if that regresses.

Run:  PYTHONPATH=. pytest -q tests/test_graph_window_vs_lifetime.py
"""
import datetime as dt

import app.services.graph_engine as graph_engine_module
from app.services.graph_engine import MultiCloudGraphEngine
from app.services.risk_engine import RiskEngine


def _event(cloud, ts, principal="attacker.x"):
    return {
        "principal": principal, "ua_family": "curl", "ua_version": "7.88.1",
        "is_known_proxy_or_tor": True, "principal_type": "IAMUser",
        "source_ip": "185.220.101.47", "source_cloud": cloud,
        "timestamp": ts, "anomaly_score": 0.9,
    }


def test_multiplier_uses_lifetime_footprint_not_window(monkeypatch):
    graph = MultiCloudGraphEngine()
    risk = RiskEngine()

    clock = {"now": 0.0}
    monkeypatch.setattr(graph_engine_module.time, "time", lambda: clock["now"])

    events = [
        _event("AWS", "2026-06-04T02:00:00Z"),
        _event("GCP", "2026-06-04T02:02:00Z"),  # +120s, outside the 60s window
    ]
    result = None
    active = []
    for e in events:
        clock["now"] = dt.datetime.fromisoformat(
            e["timestamp"].replace("Z", "+00:00")).timestamp()
        entity_id, active, _new, _method, lifetime_clouds = graph.process_event(e)
        result = risk.calculate_intensity(
            active_events=active, lifetime_clouds=lifetime_clouds,
            entity_id=entity_id, principal_type=e["principal_type"])

    # The 60s window holds ONLY the 2nd event...
    assert len(active) == 1
    # ...yet the lifetime footprint (and thus the multiplier) spans both clouds.
    assert result["cloud_span_count"] == 2
    assert result["novel_cloud_span_count"] == 1
    assert result["diversity_multiplier"] > 1.0
