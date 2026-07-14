# app/services/narrative_cache.py
"""
In-process narrative cache for FaithfulnessGatedXAI.

Repeated alerts in a live SOC feed frequently share the same predicted
ATT&CK phase and the same dominant SHAP features (e.g. ten "Impossible
Travel + Rare ASN" credential-access alerts from the same campaign). Rather
than re-calling the LLM for each one, we key on a *signature* of the alert
(phase + sorted top-SHAP feature names + a coarse confidence bucket) instead
of the raw event, so near-identical alerts reuse a narrative.

This is a plain in-process LRU dict, not a distributed cache — it resets on
restart and is per-process. That's a deliberate scope choice: it's cheap,
requires no new infrastructure, and covers the common case (bursts of
similar alerts within one running process). If this needs to survive
restarts or be shared across multiple app instances later, back it with the
same Postgres/SQLite connection the rest of the app already uses instead of
inventing a second cache technology.
"""

import hashlib
from collections import OrderedDict
from typing import Optional

_MAX_ENTRIES = 2000
_cache: "OrderedDict[str, str]" = OrderedDict()


def make_signature(predicted_phase: str, shap_attributions: dict, confidence: float) -> str:
    """
    Build a cache key from the *shape* of an alert, not its exact contents.

    Deliberately excludes entity/timestamp/cloud so that alerts which are
    substantively the same (same phase, same drivers, similar confidence)
    share a cache entry. The caller is responsible for making sure the
    surrounding dashboard context (entity, cloud, time) still reflects the
    real event even when the narrative text itself was reused.
    """
    top_features = sorted(shap_attributions.keys())
    confidence_bucket = round(confidence, 1)  # 0.94 and 0.89 both bucket to 0.9
    raw = f"{predicted_phase}|{top_features}|{confidence_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get(signature: str) -> Optional[str]:
    if signature in _cache:
        _cache.move_to_end(signature)  # touch for LRU
        return _cache[signature]
    return None


def put(signature: str, narrative: str) -> None:
    _cache[signature] = narrative
    _cache.move_to_end(signature)
    if len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)  # evict least-recently-used


def stats() -> dict:
    return {"entries": len(_cache), "max_entries": _MAX_ENTRIES}


def clear() -> None:
    """Mainly for tests."""
    _cache.clear()