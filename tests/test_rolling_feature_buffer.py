"""
Phase 5 tests: the stateful per-user rolling-feature buffer.

The headline test proves train/serve parity for rolling features: the SAME
10-event user sequence, scored (a) one event at a time through the buffer and
(b) as one batch, must yield IDENTICAL engineered rows. That single equality is
the whole point of the buffer -- live per-event scoring == offline batch scoring
for velocity/scope features.

Run:  PYTHONPATH=. pytest -q tests/test_rolling_feature_buffer.py
"""
import datetime as dt

import numpy as np
import pytest

from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from app.services.event_history import EventHistoryBuffer
from models.scoring_utils import featurize_and_align


def _event(user, ts, action, status="SUCCESS", ip="203.0.113.10", cloud="AWS"):
    return {
        "timestamp": ts, "source_cloud": cloud, "event_type": "api",
        "user_id": user, "source_ip": ip, "destination_ip": None,
        "resource": f"res-{action}", "action": action, "status": status,
        "severity": "LOW", "mfa_authenticated": True,
        "device_compliant_status": "Compliant", "user_agent": "curl/7.88.1",
        "geo_country": "US", "account_type": "USER", "principal_type": "IAMUser",
        "principal_created_in_window": False, "is_known_proxy_or_tor": "False",
        "ua_family": "curl", "ua_version": "7.88.1",
    }


def _sequence(user="alice.chen"):
    base = dt.datetime(2026, 6, 4, 9, 0, 0, tzinfo=dt.timezone.utc)
    # A mix of tight bursts (to move api_call_count_1m / error_rate_5m) and
    # gaps (to exercise the trailing windows), varied IPs/actions/statuses.
    offsets_actions = [
        (0, "ConsoleLogin", "SUCCESS", "203.0.113.10"),
        (20, "ConsoleLogin", "FAILURE", "203.0.113.10"),
        (40, "ConsoleLogin", "FAILURE", "203.0.113.11"),
        (70, "ListUsers", "SUCCESS", "203.0.113.11"),
        (300, "GetUser", "SUCCESS", "203.0.113.12"),
        (600, "CreateUser", "SUCCESS", "203.0.113.12"),
        (900, "AttachUserPolicy", "SUCCESS", "203.0.113.13"),
        (1500, "AssumeRole", "SUCCESS", "203.0.113.14"),
        (3600, "ListRoles", "SUCCESS", "203.0.113.15"),
        (7200, "DescribeInstances", "SUCCESS", "203.0.113.16"),
    ]
    evs = []
    for off, action, status, ip in offsets_actions:
        ts = (base + dt.timedelta(seconds=off)).strftime("%Y-%m-%dT%H:%M:%SZ")
        evs.append(_event(user, ts, action, status, ip))
    return evs


def test_buffer_matches_batch_engineered_rows():
    events = _sequence()

    # (b) batch: fit encoders + featurize all 10 at once, aligned to input order.
    # featurize_and_align expects __eval_row_id__ on each event (normally set by
    # load_normalize_and_label); set it here since we hand-build events.
    batch_extractor = MLFeatureExtractor()
    X_batch = featurize_and_align(
        [dict(e, __eval_row_id__=i) for i, e in enumerate(events)], batch_extractor,
        feature_columns=None, is_training=True, include_labeled_only_features=True,
    )
    cols = list(X_batch.columns)

    # (a) incremental: reuse the SAME fitted encoders (is_training=False) and
    # feed events one at a time through the buffer, recovering the current row.
    buf = EventHistoryBuffer()
    inc_extractor = MLFeatureExtractor()
    inc_extractor.label_encoders = batch_extractor.label_encoders

    for i, e in enumerate(events):
        history = buf.add_and_snapshot(e)
        batch_with_ids = [dict(h, __eval_row_id__=j) for j, h in enumerate(history)]
        current_id = len(batch_with_ids) - 1
        X, _ = inc_extractor.extract_features(
            batch_with_ids, is_training=False, include_labeled_only_features=True)
        row = X[X["__eval_row_id__"] == current_id].reindex(columns=cols, fill_value=0)

        expected = X_batch.iloc[[i]].reset_index(drop=True)
        got = row.reset_index(drop=True)
        assert np.allclose(got.to_numpy(dtype=float), expected.to_numpy(dtype=float)), (
            f"event {i} ({events[i]['action']}): buffer row != batch row\n"
            f"diff cols: {[c for c in cols if not np.isclose(float(got[c][0]), float(expected[c][0]))]}"
        )

    # Sanity: the rolling features actually MOVED across the sequence (otherwise
    # the parity above would be trivially true on constant columns).
    assert X_batch["api_call_count_1m"].max() > X_batch["api_call_count_1m"].min()
    assert X_batch["unique_ips_last_24h"].max() > 1


def test_time_eviction_bounds_history():
    buf = EventHistoryBuffer(max_window_seconds=3600)  # 1h window
    user = "bob"
    base = dt.datetime(2026, 6, 4, 9, 0, 0, tzinfo=dt.timezone.utc)
    for mins in [0, 30, 90, 150]:  # last event is 150m; only >=90m are within 1h
        ts = (base + dt.timedelta(minutes=mins)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist = buf.add_and_snapshot(_event(user, ts, "GetUser"))
    # events at 90 and 150 minutes are within 1h of 150; 0 and 30 evicted.
    assert len(hist) == 2


def test_per_user_and_global_caps():
    buf = EventHistoryBuffer(per_user_cap=3, max_users=2)
    base = dt.datetime(2026, 6, 4, 9, 0, 0, tzinfo=dt.timezone.utc)
    for i in range(5):
        ts = (base + dt.timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist = buf.add_and_snapshot(_event("u1", ts, "GetUser"))
    assert len(hist) == 3  # per-user cap holds

    # Add two more distinct users -> u1 (least recently used) is evicted.
    buf.add_and_snapshot(_event("u2", "2026-06-04T10:00:00Z", "GetUser"))
    buf.add_and_snapshot(_event("u3", "2026-06-04T10:00:01Z", "GetUser"))
    assert buf.stats()["tracked_users"] == 2
