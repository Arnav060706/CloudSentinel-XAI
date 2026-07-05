# app/services/risk_engine.py
"""
RiskEngine — Simple Decayed Cross-Cloud Risk Score
==================================================

This is a deliberately SIMPLE risk score. It is NOT a Hawkes point process
(an earlier version was; that framing, plus maximum-likelihood fitting and a
goodness-of-fit test, was removed to keep the engine easy to understand and
easy to defend). If a reviewer or advisor later wants the point-process
version back, it lives in the project history.

THE WHOLE MODEL IN THREE PLAIN RULES
------------------------------------
  1. Every recent event adds risk. How much = how anomalous it is
     (the anomaly_score from the Isolation Forest, a number in [0, 1]).
  2. Old risk fades. An event's weight HALVES every `half_life_seconds`.
     So a fresh event counts fully; one that is `half_life` old counts half;
     one that is two half-lives old counts a quarter; and so on.
  3. Crossing clouds multiplies the total. Each NEW cloud an identity touches
     (beyond what is normal for that identity) multiplies the risk by
     `per_cloud_multiplier`.

The formula is literally:

    risk = baseline
         + cloud_multiplier * Σ_i ( anomaly_i * 0.5 ** (age_i / half_life) )

    cloud_multiplier = per_cloud_multiplier ** (number of NEW clouds crossed)

Then we squash risk into a 0..1 "score" with a simple saturating curve
(risk / (risk + 1)) and raise a critical alert when the score crosses a
threshold. No sigmoids to calibrate, no exponentials to fear — just a
weighted, time-decayed sum with a cross-cloud multiplier.

WHY "NEW clouds" and not just "all clouds"?
-------------------------------------------
Some legitimate identities are normally multi-cloud (DevOps engineers, CI/CD
service accounts, multi-cloud tools). Multiplying THEM up would cause false
alarms. So the multiplier only counts clouds an identity touches BEYOND its
learned-normal set (its "baseline"). If we have no baseline for an identity
yet, we fall back to counting clouds beyond the first — which still means a
single-cloud identity is never amplified.

WHY human vs automation?
------------------------
Automation is expected to be cross-cloud, so automation principals use a
gentler `automation_cloud_multiplier`. A human suddenly going cross-cloud is
more suspicious than a pipeline that always was.

Pipeline integration (unchanged from before)
---------------------------------------------
MultiCloudGraphEngine.process_event() returns:
    (entity_id, active_events, is_new, method, lifetime_clouds)
Call:
    risk = engine.calculate_intensity(
        active_events, lifetime_clouds=lifetime_clouds,
        entity_id=entity_id, principal_type=<optional>)

Output dict keeps every key earlier code depends on (risk_intensity,
scaled_score, cloud_span_count, active_clouds, diversity_multiplier,
is_critical, dominant_signal, event_contributions, ...), so ingest.py,
db_flusher.py and metrics_exporter.py need no changes.
"""

import time
import math
import datetime
import logging
from typing import Dict, List, Optional, Set, Iterable

logger = logging.getLogger(__name__)


def _finite(x, default=0.0):
    """Return x as a finite float, or `default` for None/NaN/inf/garbage."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return default
    return xf if math.isfinite(xf) else default


class RiskEngine:
    """
    Simple time-decayed cross-cloud risk score.

    Parameters
    ----------
    baseline : float
        Small constant risk of a quiet environment. Default 0.05.
    half_life_seconds : float
        An event's weight halves every this many seconds. Smaller = risk fades
        faster. Default 30.0 (weight halves every 30s).
    per_cloud_multiplier : float
        Risk is multiplied by this for each NEW cloud a (human) identity
        crosses beyond its normal set. Default 3.0
        (0 new clouds -> ×1, 1 -> ×3, 2 -> ×9).
    automation_cloud_multiplier : float
        Gentler per-cloud multiplier for automation/service principals, which
        are expected to be cross-cloud. Default 1.7.
    alert_threshold : float
        Scaled score (0..1) at or above which is_critical is True.
        Default 0.75 (i.e. raw risk >= 3.0). Set from benign data with
        recalibrate_threshold().
    """

    _AUTOMATION_PRINCIPAL_TYPES: Set[str] = {
        "SERVICEACCOUNT", "ASSUMEDROLE", "ROLE", "FEDERATEDUSER",
        "AWSSERVICE", "AWSACCOUNT",
    }
    _AUTOMATION_UA_HINTS = (
        "boto", "aws-cli", "aws-sdk", "terraform", "botocore", "azure-sdk",
        "azure-cli", "google-api", "gcloud", "kubectl", "curl", "powershell",
        "python-requests", "go-http", "okhttp",
    )

    def __init__(
        self,
        baseline: float = 0.05,
        half_life_seconds: float = 30.0,
        per_cloud_multiplier: float = 3.0,
        automation_cloud_multiplier: float = 1.7,
        alert_threshold: float = 0.75,
    ):
        if baseline < 0:
            raise ValueError("baseline must be non-negative")
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        if per_cloud_multiplier < 1:
            raise ValueError("per_cloud_multiplier must be >= 1")
        if not (1 <= automation_cloud_multiplier <= per_cloud_multiplier):
            raise ValueError("automation_cloud_multiplier must be in [1, per_cloud_multiplier]")

        self.baseline = float(baseline)
        self.half_life = float(half_life_seconds)
        self.per_cloud_multiplier = float(per_cloud_multiplier)
        self.automation_cloud_multiplier = float(automation_cloud_multiplier)
        self.alert_threshold = float(alert_threshold)

        # Learned "normal" cloud footprint per identity: entity_key -> set(clouds).
        self._entity_baselines: Dict[str, Set[str]] = {}

    # ================================================================== #
    # Main entry point                                                   #
    # ================================================================== #

    def calculate_intensity(
        self,
        active_events: List[dict],
        lifetime_clouds: Optional[Iterable[str]] = None,
        eval_time: Optional[float] = None,
        entity_id: Optional[str] = None,
        principal_type: Optional[str] = None,
    ) -> dict:
        """
        Compute the risk score for one entity from its recent events.

        (The method name is kept as `calculate_intensity` so the rest of the
        pipeline doesn't need changes.)
        """
        # When did we evaluate "now"? Default to the newest event's time so
        # replaying historical data works the same as live (no wall-clock skew).
        parsed = [p for p in (self._parse_ts(e) for e in active_events) if p is not None]
        if eval_time is not None:
            now = _finite(eval_time, time.time())
        elif parsed:
            now = max(parsed)
        else:
            now = time.time()

        if not active_events:
            return self._empty_result()

        entity_key = self._entity_key(entity_id, active_events)
        pclass = self._classify_principal(principal_type, active_events)

        # --- Which clouds has this identity touched? ----------------------
        window_clouds = {self._normalize_cloud(e.get("source_cloud")) for e in active_events}
        window_clouds.discard("UNKNOWN")

        if lifetime_clouds is not None:
            lifetime = {self._normalize_cloud(c) for c in lifetime_clouds}
            lifetime.discard("UNKNOWN")
        else:
            lifetime = set(window_clouds)
        if not lifetime:
            lifetime = {"UNKNOWN"}

        # --- Rule 3: how many NEW clouds beyond this identity's normal? ----
        baseline_clouds = self._entity_baselines.get(entity_key)
        if baseline_clouds:
            novel_count = len(lifetime - baseline_clouds)
        else:
            novel_count = max(0, len(lifetime) - 1)  # first cloud is "free"

        per_cloud = (self.per_cloud_multiplier if pclass == "human"
                     else self.automation_cloud_multiplier)
        cloud_multiplier = per_cloud ** novel_count

        # --- Rules 1 & 2: time-decayed sum of anomaly scores --------------
        decayed_sum = 0.0
        contributions = []
        for e in active_events:
            t_i = self._parse_ts(e)
            if t_i is None:
                logger.warning("Event missing/invalid 'timestamp' — skipping.")
                continue
            age = now - t_i
            if age < 0:
                age = 0.0  # future event (clock skew) counts as "now"

            anomaly = e.get("anomaly_score")
            if anomaly is None:
                anomaly = e.get("phase_confidence", 0.1)
            anomaly = min(1.0, max(0.0, _finite(anomaly, 0.1)))

            weight = 0.5 ** (age / self.half_life)   # halves every half_life seconds
            contribution = anomaly * weight
            decayed_sum += contribution
            contributions.append({
                "principal": e.get("principal", ""),
                "source_cloud": self._normalize_cloud(e.get("source_cloud")),
                "anomaly_score": round(anomaly, 4),
                "age_seconds": round(age, 2),
                "weight": round(weight, 4),
                "contribution": round(contribution, 6),
            })

        # --- Combine ------------------------------------------------------
        risk = self.baseline + cloud_multiplier * decayed_sum
        risk = _finite(risk, self.baseline)

        # Squash to 0..1 (saturating curve; approaches 1 as risk grows).
        scaled = risk / (risk + 1.0)
        is_critical = scaled >= self.alert_threshold

        return {
            "risk_intensity": round(risk, 6),
            "scaled_score": round(scaled, 4),
            "cloud_span_count": len(lifetime) if lifetime != {"UNKNOWN"} else 0,
            "window_cloud_span_count": len(window_clouds),
            "novel_cloud_span_count": novel_count,
            "baseline_cloud_span_count": len(baseline_clouds) if baseline_clouds else 0,
            "active_clouds": sorted(c for c in lifetime if c != "UNKNOWN"),
            "diversity_multiplier": round(cloud_multiplier, 4),
            "principal_class": pclass,
            "is_critical": bool(is_critical),
            "event_contributions": contributions,
            "dominant_signal": self._dominant_signal(risk, novel_count, decayed_sum),
        }

    # ================================================================== #
    # Optional helpers (simple, not required to run)                     #
    # ================================================================== #

    def record_baseline(self, entity_id: str, clouds: Iterable[str]) -> None:
        """
        Tell the engine which clouds are NORMAL for an identity, so operating
        in them is not treated as risky. Additive; call during a benign pass
        or seed it from known DevOps/service accounts.
        """
        norm = {self._normalize_cloud(c) for c in clouds}
        norm.discard("UNKNOWN")
        if not norm:
            return
        self._entity_baselines.setdefault(str(entity_id), set()).update(norm)

    def recalibrate_threshold(self, benign_risk_values: List[float],
                              percentile: float = 99.0) -> float:
        """
        Set alert_threshold from benign traffic: take the given percentile of
        the benign SCALED scores. Run known-benign data through the engine,
        collect the risk_intensity values, and pass them here.
        Returns the new alert_threshold.
        """
        if not benign_risk_values:
            raise ValueError("benign_risk_values must be non-empty")
        scaled = sorted((r / (r + 1.0)) for r in map(lambda v: _finite(v, 0.0), benign_risk_values))
        idx = min(int(len(scaled) * percentile / 100.0), len(scaled) - 1)
        self.alert_threshold = round(min(0.999, scaled[idx]), 4)
        logger.info("alert_threshold recalibrated to %.4f (p%.0f)", self.alert_threshold, percentile)
        return self.alert_threshold

    def get_parameter_summary(self) -> dict:
        return {
            "baseline": self.baseline,
            "half_life_seconds": self.half_life,
            "per_cloud_multiplier": self.per_cloud_multiplier,
            "automation_cloud_multiplier": self.automation_cloud_multiplier,
            "alert_threshold": self.alert_threshold,
            "tracked_entities": len(self._entity_baselines),
        }

    # ================================================================== #
    # Internal helpers                                                   #
    # ================================================================== #

    def _entity_key(self, entity_id, active_events) -> str:
        if entity_id:
            return str(entity_id)
        principals = sorted({str(e.get("principal", "")) for e in active_events if e.get("principal")})
        return "|".join(principals) if principals else "unknown_entity"

    def _classify_principal(self, principal_type, active_events) -> str:
        ptype = principal_type
        if ptype is None:
            counts: Dict[str, int] = {}
            for e in active_events:
                pt = str(e.get("principal_type", "") or "").upper()
                if pt:
                    counts[pt] = counts.get(pt, 0) + 1
            ptype = max(counts, key=counts.get) if counts else ""
        if str(ptype or "").upper() in self._AUTOMATION_PRINCIPAL_TYPES:
            return "automation"
        for e in active_events:
            ua = str(e.get("ua_family", "") or e.get("user_agent", "")).lower()
            if any(h in ua for h in self._AUTOMATION_UA_HINTS):
                return "automation"
        return "human"

    def _normalize_cloud(self, raw) -> str:
        c = str(raw or "UNKNOWN").upper()
        return "AZURE" if c == "ENTRA-ID" else c

    def _parse_ts(self, event) -> Optional[float]:
        ts = event.get("timestamp")
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return float(ts) if math.isfinite(float(ts)) else None
        try:
            return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            logger.warning("Could not parse timestamp '%s'", ts)
            return None

    def _dominant_signal(self, risk, novel_count, decayed_sum) -> str:
        if risk <= self.baseline * 1.5:
            return "baseline_only"
        if novel_count >= 1:
            return "cross_cloud_diversity"
        if decayed_sum > self.baseline * 5:
            return "high_event_volume"
        return "recent_activity_accumulation"

    def _empty_result(self) -> dict:
        return {
            "risk_intensity": round(self.baseline, 6),
            "scaled_score": round(self.baseline / (self.baseline + 1.0), 4),
            "cloud_span_count": 0,
            "window_cloud_span_count": 0,
            "novel_cloud_span_count": 0,
            "baseline_cloud_span_count": 0,
            "active_clouds": [],
            "diversity_multiplier": 1.0,
            "principal_class": "unknown",
            "is_critical": False,
            "event_contributions": [],
            "dominant_signal": "baseline_only",
        }


# Backward-compatibility alias. The class is no longer a Hawkes process, but
# keep the old name importable so nothing breaks if a reference is missed.
HawkesRiskEngine = RiskEngine