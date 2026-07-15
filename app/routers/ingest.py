# app/routers/ingest.py
"""
Telemetry Ingestion Pipeline Router
===================================

Two ingestion surfaces:

  POST /api/v1/ingest       — accepts ALREADY-NORMALIZED UnifiedLogModel
                              records (the original contract).
  POST /api/v1/ingest/raw   — accepts RAW multi-cloud provider logs (AWS
                              CloudTrail / Azure / GCP JSON) and runs them
                              through ParserPipeline first. This is what
                              makes the service genuinely end-to-end with
                              the mock_data you already have.

Both hand each normalized event to the same background analytical chain:

    1. Parallel ML inference (Isolation Forest anomaly + XGBoost phase)
    2. Identity stitching & lifetime cross-cloud tracking (graph engine)
    3. Hawkes self-exciting intensity, driven by the entity's LIFETIME
       cloud footprint (pace-independent cross-cloud detection)
    4. Faithfulness-gated LLM narrative on critical alerts
    5. Write-coalesced risk-state persistence

Returns 202 immediately and offloads compute to BackgroundTasks.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, BackgroundTasks, status

import app.main as main_core
from app.parser_normalizer.src.schema import UnifiedLogModel
from app.core.database import XAIAlert, AsyncSessionLocal
from app.services.db_flusher import pending_risk_updates
from app.services import metrics_exporter
from app.services.loki_exporter import push_narrative_to_loki

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telemetry Ingestion Pipeline"])

# ParserPipeline is only needed for the raw endpoint; import lazily so a
# missing GeoLite DB / optional dep can't take down the whole router import.
_parser_pipeline = None


def _get_parser_pipeline():
    global _parser_pipeline
    if _parser_pipeline is None:
        from app.parser_normalizer.src.pipeline import ParserPipeline
        _parser_pipeline = ParserPipeline()
    return _parser_pipeline


def _engines_ready() -> bool:
    sm = getattr(main_core, "state_matrix", {})
    return "graph_engine" in sm and "risk_engine" in sm


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_multi_cloud_stream(
    payload: List[UnifiedLogModel],
    background_tasks: BackgroundTasks,
):
    """Ingest already-normalized cross-cloud logs."""
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty telemetry payload array.",
        )

    if not _engines_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics engines not initialized yet.",
        )

    for log in payload:
        log_dict = log.model_dump()
        # Pydantic gives us datetime / IPvAnyAddress objects; the engines
        # expect plain strings. Normalize the couple of fields they touch.
        _stringify_engine_fields(log_dict)
        background_tasks.add_task(process_log_through_engine, log_dict)

    return {"status": "Telemetry Accepted", "records_processed": len(payload)}


@router.post("/ingest/raw", status_code=status.HTTP_202_ACCEPTED)
async def ingest_raw_multi_cloud_stream(
    payload: List[dict],
    background_tasks: BackgroundTasks,
):
    """Ingest RAW provider logs; normalize via ParserPipeline, then analyze."""
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty raw telemetry payload array.",
        )

    if not _engines_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics engines not initialized yet.",
        )

    pipeline = _get_parser_pipeline()
    accepted = 0
    failed = 0
    for raw_log in payload:
        normalized = pipeline.process_log(raw_log)
        if normalized is None:
            failed += 1
            continue
        log_dict = normalized.model_dump()
        _stringify_engine_fields(log_dict)
        background_tasks.add_task(process_log_through_engine, log_dict)
        accepted += 1

    return {
        "status": "Telemetry Accepted",
        "records_normalized": accepted,
        "records_failed": failed,
    }


def _stringify_engine_fields(log_dict: dict) -> None:
    """
    The graph + risk engines key off `principal`, `timestamp`, `source_ip`
    and `source_cloud` as plain strings. Normalized records carry a
    datetime timestamp, an IP object, and use `user_id` for the principal.
    Bridge those here so both endpoints feed the engines a consistent shape.
    """
    # principal: the engines look for "principal"; normalized logs use user_id
    if "principal" not in log_dict:
        log_dict["principal"] = log_dict.get("user_id", "unknown_principal")
    # timestamp -> ISO string
    ts = log_dict.get("timestamp")
    if ts is not None and not isinstance(ts, str):
        log_dict["timestamp"] = ts.isoformat()
    # source_ip -> str (or None)
    ip = log_dict.get("source_ip")
    if ip is not None and not isinstance(ip, str):
        log_dict["source_ip"] = str(ip)


async def process_log_through_engine(log_data: dict):
    """Background execution chain running the analytical layers in sequence."""
    entity_id = "unknown_principal"
    try:
        sm = main_core.state_matrix
        graph = sm["graph_engine"]
        hawkes_engine = sm["risk_engine"]
        ml_engine = sm["ml_engine"]          # shared singleton
        xai_engine = sm["xai_engine"]        # shared singleton

        # -- Step 1: Parallel ML inference (or neutral defaults in bypass) --
        scored_log = await ml_engine.execute_parallel_inference(log_data)

        # Prometheus: every event processed + latest per-engine risk gauges.
        metrics_exporter.record_event_processed()
        metrics_exporter.record_ml_scores(
            cloud_provider=scored_log.get("source_cloud", "UNKNOWN"),
            anomaly_score=scored_log.get("anomaly_score", 0.0),
            phase_confidence=scored_log.get("phase_confidence", 0.0),
        )
        metrics_exporter.record_phase_and_shap(
            cloud_provider=scored_log.get("source_cloud", "UNKNOWN"),
            predicted_phase=scored_log.get("predicted_phase", "Normal"),
            shap_attributions=scored_log.get("shap_attributions", {}),
        )

        # -- Step 2: Identity stitching + lifetime cross-cloud tracking -----
        # New graph engine returns a 5-tuple; lifetime_clouds is what makes
        # cross-cloud detection pace-independent.
        entity_id, active_events, is_new, method, lifetime_clouds = graph.process_event(scored_log)

        if not active_events:
            logger.debug("Empty risk window for entity %s (method=%s)", entity_id, method)
            return

        # -- Step 3: Hawkes intensity, driven by LIFETIME cloud footprint ---
        risk_result = hawkes_engine.calculate_intensity(
            active_events,
            lifetime_clouds=lifetime_clouds,
        )

        # Prometheus: trust score / risk intensity / cloud span / criticals.
        metrics_exporter.record_risk_result(
            user_identity=scored_log.get("principal", entity_id),
            cloud_provider=scored_log.get("source_cloud", "UNKNOWN"),
            risk_result=risk_result,
        )

        # -- Step 4: Faithfulness gate + LLM narrative on critical alerts ---
        if risk_result.get("is_critical", False):
            logger.warning(
                "[CRITICAL THREAT] entity=%s intensity=%s clouds=%s",
                entity_id, risk_result.get("risk_intensity"),
                risk_result.get("active_clouds"),
            )
            passed_gate, narrative, generation_ok = await xai_engine.generate_forensic_narrative(
                scored_log, risk_result
            )

            # Push a REAL log line to Loki for the dashboard's logs panel.
            # If the LLM produced a narrative, stream that; otherwise stream a
            # structured summary of the actual finding (still real, not faked).
            loki_line = narrative if (passed_gate and generation_ok) else (
                f"phase={scored_log.get('predicted_phase', 'Unknown')} "
                f"dominant_signal={risk_result.get('dominant_signal')} "
                f"clouds={risk_result.get('active_clouds')} "
                f"scaled_score={risk_result.get('scaled_score')} "
                f"(narrative unavailable: gate={passed_gate}, llm_ok={generation_ok})"
            )
            await push_narrative_to_loki(
                entity_id=entity_id,
                cloud_provider=scored_log.get("source_cloud", "UNKNOWN"),
                action=scored_log.get("action", "Unknown"),
                risk_score=risk_result.get("scaled_score"),
                narrative_text=loki_line,
                phase=scored_log.get("predicted_phase", "Normal"),
                shap_attributions=scored_log.get("shap_attributions", {}),
                principal=scored_log.get("principal", entity_id),
            )

            # Only persist a real, successfully generated narrative to the
            # permanent ledger. A failed/gated generation is NOT written as
            # if it were a forensic finding (previous bug).
            if passed_gate and generation_ok:
                logger.info("SOC narrative for %s: %s", entity_id, narrative)
                async with AsyncSessionLocal() as session:
                    try:
                        alert = XAIAlert(
                            entity_id=entity_id,
                            predicted_phase=scored_log.get("predicted_phase", "Unknown"),
                            dominant_shap_signal=risk_result.get("dominant_signal", "baseline_only"),
                            llama_narrative=narrative,
                        )
                        session.add(alert)
                        await session.commit()
                    except Exception as db_err:
                        logger.error("Failed to commit XAI narrative: %s", db_err)
                        await session.rollback()
            else:
                logger.info(
                    "Narrative not persisted for %s (passed_gate=%s, generation_ok=%s): %s",
                    entity_id, passed_gate, generation_ok, narrative,
                )

        # -- Step 5: Write-coalesced risk-state buffer ----------------------
        pending_risk_updates[entity_id] = risk_result

    except Exception as pipeline_error:
        logger.error(
            "Fatal pipeline breakdown for entity '%s': %s",
            entity_id, pipeline_error, exc_info=True,
        )