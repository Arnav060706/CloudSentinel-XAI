# app/services/metrics_exporter.py
"""
Prometheus metrics for the REAL pipeline.

These are updated from inside the ingestion pipeline (app/routers/ingest.py),
so what Prometheus scrapes reflects genuine detector output — not the random
values the old demo simulator produced. The metric NAMES intentionally match
the ones grafana.json already queries, so the existing dashboard works with
no edits:

    security_pipeline_processed_events_total   (Counter)
    security_engine_risk_probability           (Gauge, per engine+cloud)
    security_user_trust_score                  (Gauge, per identity+cloud)

Plus a few extra real signals you can add panels for later.

CARDINALITY NOTE: `security_user_trust_score` is labeled by user_identity.
In a bounded lab dataset that's fine. In production, principal strings are
high-cardinality and Prometheus is the wrong place for per-identity series —
move per-entity detail to Loki/SQLite and keep only per-cloud aggregates here.
"""

from prometheus_client import Counter, Gauge

PIPELINE_EVENTS = Counter(
    "security_pipeline_processed_events_total",
    "Total multi-cloud log events processed by the core pipeline",
)

ENGINE_RISK = Gauge(
    "security_engine_risk_probability",
    "Latest risk/anomaly probability output by each analytical engine",
    ["engine_name", "cloud_provider"],
)

USER_TRUST = Gauge(
    "security_user_trust_score",
    "Trust score (100 = fully trusted) derived from entity risk (100*(1-scaled_score))",
    ["user_identity", "cloud_provider"],
)

# ---- Extra real signals (not required by the current dashboard) ---------
MAX_RISK_INTENSITY = Gauge(
    "security_max_risk_intensity",
    "Highest raw Hawkes lambda intensity observed in the most recent evaluation",
)

CRITICAL_ALERTS = Counter(
    "security_critical_alerts_total",
    "Total critical alerts raised, by cloud provider",
    ["cloud_provider"],
)

ENTITY_CLOUD_SPAN = Gauge(
    "security_entity_cloud_span",
    "Lifetime distinct-cloud footprint size for the most recently scored entity",
)


def record_event_processed() -> None:
    PIPELINE_EVENTS.inc()


def record_ml_scores(cloud_provider: str, anomaly_score: float, phase_confidence: float) -> None:
    cloud = (cloud_provider or "UNKNOWN").upper()
    ENGINE_RISK.labels(engine_name="Isolation_Forest", cloud_provider=cloud).set(float(anomaly_score))
    ENGINE_RISK.labels(engine_name="XGBoost", cloud_provider=cloud).set(float(phase_confidence))


def record_risk_result(user_identity: str, cloud_provider: str, risk_result: dict) -> None:
    cloud = (cloud_provider or "UNKNOWN").upper()
    scaled = float(risk_result.get("scaled_score", 0.0))
    trust = max(0.0, min(100.0, 100.0 * (1.0 - scaled)))
    USER_TRUST.labels(user_identity=str(user_identity), cloud_provider=cloud).set(trust)
    MAX_RISK_INTENSITY.set(float(risk_result.get("risk_intensity", 0.0)))
    ENTITY_CLOUD_SPAN.set(int(risk_result.get("cloud_span_count", 0)))
    if risk_result.get("is_critical"):
        CRITICAL_ALERTS.labels(cloud_provider=cloud).inc()