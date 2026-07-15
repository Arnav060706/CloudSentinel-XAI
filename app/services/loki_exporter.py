# app/services/loki_exporter.py
"""
Pushes REAL forensic narratives from the pipeline into Loki, on the same
stream label (`job="security-triage-pipeline"`) that the Grafana dashboard's
logs panel already queries. Non-blocking and best-effort: a Loki outage must
never break the analytical pipeline.

Label hygiene: only low-cardinality labels (job, cloud_provider) are used as
Loki stream labels. Everything else (entity id, action, risk score) goes in
the log LINE, not the labels — keeping Loki's index small.
"""

import os
import time
import json
import logging

logger = logging.getLogger(__name__)

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

LOKI_BASE = os.environ.get("LOKI_URL", "http://localhost:3100")
LOKI_PUSH_URL = LOKI_BASE.rstrip("/") + "/loki/api/v1/push"


async def push_narrative_to_loki(
    entity_id: str,
    cloud_provider: str,
    action: str,
    risk_score,
    narrative_text: str,
    phase: str = "Normal",
    shap_attributions: dict = None,
    principal: str = None,
) -> None:
    if not _HTTPX_AVAILABLE:
        logger.debug("httpx not installed — skipping Loki push.")
        return

    timestamp_ns = str(time.time_ns())
    # Sanitize newlines/quotes so the log line stays single-line and clean.
    clean = str(narrative_text).replace("\n", " ").replace('"', "'").strip()
    shap_json = json.dumps(shap_attributions or {}, default=str)
    line = (
        f"[CRITICAL XAI REPORT] entity={entity_id} principal={principal or entity_id} "
        f"action={action} risk={risk_score} phase={phase} :: {clean} :: shap={shap_json}"
    )

    payload = {
        "streams": [
            {
                "stream": {
                    "job": "security-triage-pipeline",
                    "cloud_provider": str(cloud_provider or "UNKNOWN").upper(),
                    "phase": str(phase or "Normal"),
                },
                "values": [[timestamp_ns, line]],
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(
                LOKI_PUSH_URL, json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code not in (200, 204):
                logger.warning("Loki push failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:  # network/connection errors are non-fatal
        logger.warning("Loki push error: %s", e)