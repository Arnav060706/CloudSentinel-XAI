# app/routers/ingest.py
"""
Telemetry Ingestion Pipeline Router
==================================

Interceptors incoming multi-cloud streaming logs from collectors or simulation 
harnesses (e.g., Stratus Red Team) and orchestrates the core analytical pipeline:

    1. Parallel ML Inference (Isolation Forest Anomaly + XGBoost ATT&CK Phase)
    2. Identity Stitching & Graph Stateful Tracking (Sliding Window Management)
    3. Hawkes Self-Exciting Point Process Evaluation (Dynamic Risk Intensity)
    4. Write-Coalescing Buffer Queuing (Asynchronous SQLite/Postgres Storage)

Design Pattern
--------------
Returns an immediate HTTP 202 Accepted response to the upstream broker/client.
The entire compute-intensive pipeline is safely offloaded to FastAPI's 
BackgroundTasks to maintain extreme ingestion throughput.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

import app.main as main_core
from app.parser_normalizer.src.schema import UnifiedLogModel
from app.services.ml_inference import ParallelMLEngine
from app.services.db_flusher import pending_risk_updates

from app.services.xai_engine import FaithfulnessGatedXAI
from app.core.database import XAIAlert
from sqlalchemy.dialects.sqlite import insert
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telemetry Ingestion Pipeline"])


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_multi_cloud_stream(
    payload: List[UnifiedLogModel], 
    background_tasks: BackgroundTasks
):
    """
    Ingestion interceptor mapping incoming cross-cloud logs directly into the
    asynchronous background analytics engine.
    """
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Empty telemetry payload array."
        )
    
    # Verify the core engines are correctly loaded in the global application context
    if "graph_engine" not in main_core.state_matrix or "risk_engine" not in main_core.state_matrix:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="System State Initialization Engine Failure: Analytics modules not loaded."
        )
        
    for log in payload:
        # Export the Pydantic data model cleanly into an operational dictionary
        log_dict = log.model_dump()
        
        # Offload the execution chain to background worker threads.
        # Since process_log_through_engine is an async def, FastAPI schedules it
        # optimally on the existing ASGI event loop.
        background_tasks.add_task(process_log_through_engine, log_dict)
        
    return {
        "status": "Telemetry Accepted", 
        "records_processed": len(payload)
    }


async def process_log_through_engine(log_data: dict):
    """
    The background execution chain running our analytical layers in sequence.
    """
    entity_id = "unknown_principal"
    try:
        # Extract global system engine references from the lifespan matrix
        graph = main_core.state_matrix["graph_engine"]
        hawkes_engine = main_core.state_matrix["risk_engine"]
        
        # ------------------------------------------------------------------ #
        # Step 1: Parallel Model Evaluations (Isolation Forest / XGBoost)    #
        # ------------------------------------------------------------------ #
        # We perform ML inference first so that the event tuple is fully 
        # enriched with anomaly_score and predicted_phase before being stored
        # in the graph's sliding window history.
        ml_engine = ParallelMLEngine(main_core.state_matrix)
        scored_log = await ml_engine.execute_parallel_inference(log_data)
        
        # ------------------------------------------------------------------ #
        # Step 2: Identity Stitching via Graph Layer                        #
        # ------------------------------------------------------------------ #
        # The graph engine maps correlates, and resolves aliases across cloud 
        # fabrics, returning the unified canonical entity identifier.
        entity_id, active_events, is_new = graph.process_event(scored_log)
        
        if not active_events:
            logger.debug(f"Event window empty or expired for entity: {entity_id}")
            return

        # ------------------------------------------------------------------ #
        # Step 3: Hawkes Process Intensity Scaling                           #
        # ------------------------------------------------------------------ #
        # Calculates the dynamic risk intensity and applies the cross-cloud 
        # diversity multiplier based on the state of the active window.
        risk_result = hawkes_engine.calculate_intensity(active_events)
        
        # ------------------------------------------------------------------ #
        # Step 4: SHAP Faithfulness Verification & Llama Generation         #
        # ------------------------------------------------------------------ #
        if risk_result.get("is_critical", False):
            logger.warning(
                f"[CRITICAL THREAT] Entity {entity_id} | Intensity: {risk_result['risk_intensity']}"
            )
            
            # Instantiate the Layer 5 Engine
            xai_engine = FaithfulnessGatedXAI(main_core.state_matrix)
            
            # Execute the Faithfulness Gate and LLM generation asynchronously
            passed_gate, narrative = await xai_engine.generate_forensic_narrative(scored_log, risk_result)
            
            # Log the narrative and push it to the persistent SQLite ledger
            logger.info(f"Generated SOC Narrative: {narrative}")
            
            async with AsyncSessionLocal() as session:
                try:
                    alert = XAIAlert(
                        entity_id=entity_id,
                        predicted_phase=scored_log.get("predicted_phase", "Unknown"),
                        dominant_shap_signal=risk_result.get("dominant_signal", "baseline_only"),
                        llama_narrative=narrative
                    )
                    session.add(alert)
                    await session.commit()
                except Exception as db_err:
                    logger.error(f"Failed to commit XAI narrative to ledger: {db_err}")
        
        # ------------------------------------------------------------------ #
        # Step 5: Native Storage Exporters Pushing                           #
        # ------------------------------------------------------------------ #
        # Non-blocking O(1) memory write. The background flusher ticker will 
        # batch and flush this state to SQLite every 3 seconds.
        pending_risk_updates[entity_id] = risk_result

    except Exception as pipeline_error:
        logger.error(
            f"Fatal pipeline breakdown while processing log for entity '{entity_id}': "
            f"{str(pipeline_error)}", 
            exc_info=True
        )