"""
event_history.py — Phase 5: stateful per-user rolling-feature buffer for LIVE
inference.

MLFeatureExtractor computes velocity/scope features (api_call_count_1m,
error_rate_5m, unique_ips_last_24h, privileged_actions_last_24h, ...) over a
user's RECENT event history, grouped by `user_id` inside a trailing time window
(largest = 24h). Called on a lone live event with no history, they all collapse
to first-event defaults, so live scores are NOT the scores the models were
trained on (train/serve skew). This buffer keeps a bounded trailing history per
user so the current event can be featurized WITH its recent context.

Keying: by the event's `user_id` — deliberately the SAME field
MLFeatureExtractor groups its rolling features on (and that the graph engine's
`principal` derives from), so this is not a third identity scheme. Getting this
key wrong would silently mis-window every rolling feature.

Bounding (all three, so memory is bounded under any load):
  * time     — evict events older than `max_window_seconds` (24h, the largest
               window any feature needs) relative to the newest event;
  * per-user — cap each user's deque at `per_user_cap` (default 500);
  * global   — LRU cap on the number of tracked users (`max_users`, default
               10k), evicting the least-recently-active user.
"""
import datetime as dt
import logging
from collections import OrderedDict, deque
from typing import List

logger = logging.getLogger(__name__)


def _epoch(ts) -> float:
    """Seconds since epoch from a datetime, epoch number, or ISO-8601 string."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    try:
        return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


class EventHistoryBuffer:
    """Bounded trailing per-user event history for rolling-feature continuity."""

    def __init__(self, max_window_seconds: float = 86400.0,
                 per_user_cap: int = 500, max_users: int = 10000):
        if per_user_cap < 1 or max_users < 1 or max_window_seconds <= 0:
            raise ValueError("caps must be positive")
        self.max_window_seconds = float(max_window_seconds)
        self.per_user_cap = int(per_user_cap)
        self.max_users = int(max_users)
        # user_key -> deque(events, oldest..newest). OrderedDict tracks LRU:
        # most-recently-touched user is moved to the end.
        self._buffers: "OrderedDict[str, deque]" = OrderedDict()

    @staticmethod
    def _key(event: dict) -> str:
        # Match MLFeatureExtractor's rolling groupby key exactly (user_id).
        return str(event.get("user_id", "") or "unknown")

    def add_and_snapshot(self, event: dict) -> List[dict]:
        """Append `event` to its user's history, apply all three bounds, and
        return the user's current trailing history (oldest..newest) with the
        just-added event LAST. Stores a shallow copy so later enrichment of the
        live event dict (anomaly_score, etc.) can't mutate buffered history."""
        key = self._key(event)
        dq = self._buffers.pop(key, None)
        if dq is None:
            dq = deque()
        dq.append(dict(event))

        # 1) time-based eviction, relative to the newest event just added.
        now = _epoch(event.get("timestamp"))
        while dq and (now - _epoch(dq[0].get("timestamp"))) > self.max_window_seconds:
            dq.popleft()
        # 2) per-user count cap (drop oldest).
        while len(dq) > self.per_user_cap:
            dq.popleft()

        # Reinsert at the END so this user is now most-recently-used.
        self._buffers[key] = dq

        # 3) global LRU cap on number of tracked users.
        while len(self._buffers) > self.max_users:
            evicted_key, _ = self._buffers.popitem(last=False)
            logger.debug("EventHistoryBuffer evicted LRU user %s", evicted_key)

        return list(dq)

    def stats(self) -> dict:
        return {
            "tracked_users": len(self._buffers),
            "total_buffered_events": sum(len(q) for q in self._buffers.values()),
            "max_window_seconds": self.max_window_seconds,
            "per_user_cap": self.per_user_cap,
            "max_users": self.max_users,
        }
