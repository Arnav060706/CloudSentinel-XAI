"""
Tests for app/services/narrative_cache.py.

These exercise the cache in isolation — no Ollama, no trained models, no
event loop required — matching how the rest of this project's tests run
fully in bypass mode without external dependencies.
"""

from app.services import narrative_cache


def setup_function():
    narrative_cache.clear()


def test_signature_is_order_independent():
    sig1 = narrative_cache.make_signature(
        "Credential Access",
        {"ua_family": {"raw_value": "curl"}, "action": {"raw_value": "ConsoleLogin"}},
        0.94,
    )
    sig2 = narrative_cache.make_signature(
        "Credential Access",
        {"action": {"raw_value": "ConsoleLogin"}, "ua_family": {"raw_value": "curl"}},
        0.91,  # same confidence bucket (rounds to 0.9)
    )
    assert sig1 == sig2


def test_signature_differs_on_phase():
    sig1 = narrative_cache.make_signature("Credential Access", {"action": {}}, 0.9)
    sig2 = narrative_cache.make_signature("Privilege Escalation", {"action": {}}, 0.9)
    assert sig1 != sig2


def test_signature_differs_on_confidence_bucket():
    sig1 = narrative_cache.make_signature("Credential Access", {"action": {}}, 0.94)
    sig2 = narrative_cache.make_signature("Credential Access", {"action": {}}, 0.65)
    assert sig1 != sig2


def test_cache_roundtrip():
    sig = narrative_cache.make_signature("Exfiltration", {"action": {}}, 0.87)
    assert narrative_cache.get(sig) is None
    narrative_cache.put(sig, "Data was staged and exfiltrated via GCS list calls.")
    assert narrative_cache.get(sig) == "Data was staged and exfiltrated via GCS list calls."


def test_lru_eviction():
    narrative_cache._MAX_ENTRIES = 3  # shrink for the test
    for i in range(5):
        sig = narrative_cache.make_signature(f"Phase{i}", {}, 0.5)
        narrative_cache.put(sig, f"narrative-{i}")
    stats = narrative_cache.stats()
    assert stats["entries"] <= 3