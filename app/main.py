# app/main.py
"""
CloudSentinel-XAI — FastAPI application entrypoint.

Lifespan responsibilities (all previously commented out, which is why the
service could not run end to end):

  1. Build the shared, process-wide `state_matrix` that every request-time
     engine reads from. Engines (graph, risk, ML, XAI) are instantiated
     ONCE here and reused, instead of being rebuilt on every event.
  2. Load the trained ML artifacts if they exist on disk. If they do NOT
     (e.g. before the dataset has been generated and the models trained),
     the app starts cleanly in BYPASS mode: inference returns safe neutral
     defaults and the rest of the pipeline still runs end to end. Drop the
     model files in later and the same code path upgrades automatically.
  3. Create the SQLite tables.
  4. Start the background flush ticker (write-coalescing risk-state writer)
     and a periodic identity-purge ticker for the graph engine.
"""

import os
import pickle
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.database import engine, Base
from app.services.graph_engine import MultiCloudGraphEngine
from app.services.risk_engine import RiskEngine
from app.services.ml_inference import ParallelMLEngine
from app.services.xai_engine import FaithfulnessGatedXAI
from app.services.db_flusher import risk_flush_ticker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Shared, process-wide state. Populated in the lifespan startup below and
# read by the ingest router at request time. Defined at module level (this
# was the crash: the ingest router references main.state_matrix, which used
# to never exist because the whole lifespan block was commented out).
# ---------------------------------------------------------------------- #
state_matrix: dict = {}
background_tasks_refs: dict = {}

# Where trained artifacts are expected. All optional — absence => bypass mode.
# Both are full bundles (model + fitted label_encoders + feature_columns, and
# for XGBoost also class_names) produced by models/train_isolation_forest.py
# and models/train_xgboost.py -- NOT the separate isolation_forest.pkl /
# xgboost_model.json / feature_encoder.pkl / class_names.json files an
# earlier version of this loader expected (those were never actually
# produced by anything; see app/services/ml_inference.py's module docstring
# for why one shared feature_encoder can't correctly serve both models here).
MODELS_DIR = os.environ.get("CLOUDSENTINEL_MODELS_DIR", "models")
ISO_FOREST_BUNDLE_PATH = os.path.join(MODELS_DIR, "iso_forest.pkl")
XGBOOST_BUNDLE_PATH = os.path.join(MODELS_DIR, "xgboost_classifier.pkl")


def _load_ml_artifacts(sm: dict) -> None:
    """
    Best-effort load of trained artifacts into the state matrix. Every load
    is independent and optional; any missing/unloadable artifact leaves its
    slot as None and the ML engine falls back to bypass behaviour.
    """
    if os.path.exists(ISO_FOREST_BUNDLE_PATH):
        try:
            with open(ISO_FOREST_BUNDLE_PATH, "rb") as f:
                sm["iso_forest_bundle"] = pickle.load(f)
            logger.info("Loaded Isolation Forest bundle from %s", ISO_FOREST_BUNDLE_PATH)
        except Exception as e:
            logger.error("Failed to load Isolation Forest bundle: %s", e)

    if os.path.exists(XGBOOST_BUNDLE_PATH):
        try:
            with open(XGBOOST_BUNDLE_PATH, "rb") as f:
                sm["xgboost_bundle"] = pickle.load(f)
            logger.info("Loaded XGBoost bundle from %s", XGBOOST_BUNDLE_PATH)
        except Exception as e:
            logger.error("Failed to load XGBoost bundle: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---------------------------------------------------------
    logger.info("CloudSentinel-XAI starting up...")

    # 1. Create database tables (idempotent).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured.")

    # 2. Load ML artifacts (optional — bypass mode if absent).
    _load_ml_artifacts(state_matrix)
    bypass = not (state_matrix.get("iso_forest_bundle") and state_matrix.get("xgboost_bundle"))
    if bypass:
        logger.warning(
            "Running in ML BYPASS mode: no trained iso_forest.pkl/"
            "xgboost_classifier.pkl bundles found under '%s'. Inference "
            "returns neutral defaults; the rest of the pipeline still runs "
            "end to end. Run models/train_isolation_forest.py and "
            "models/train_xgboost.py to enable real scoring.", MODELS_DIR,
        )

    # 3. Instantiate the stateful engines ONCE and share them.
    state_matrix["graph_engine"] = MultiCloudGraphEngine()
    state_matrix["risk_engine"] = RiskEngine()
    # These two read models out of state_matrix; build once, reuse per event
    # (previously they were rebuilt on every single log — rebuilding the SHAP
    # TreeExplainer and Ollama client each time).
    state_matrix["ml_engine"] = ParallelMLEngine(state_matrix)
    state_matrix["xai_engine"] = FaithfulnessGatedXAI(state_matrix)

    # 4. Background write-coalescing flush ticker.
    flush_task = asyncio.create_task(risk_flush_ticker(flush_interval_seconds=3))
    background_tasks_refs["flush_ticker"] = flush_task

    # 5. Periodic identity-purge ticker for the graph engine (bounds memory
    #    now that identity persists across idle windows — see graph_engine).
    async def _purge_ticker(interval_seconds: int = 3600, max_idle_seconds: int = 86400):
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    removed = state_matrix["graph_engine"].purge_stale_entities(max_idle_seconds)
                    if removed:
                        logger.info("Graph purge removed %d stale entities.", removed)
                except Exception as e:
                    logger.error("Graph purge failed: %s", e)
        except asyncio.CancelledError:
            logger.info("Purge ticker terminated.")

    purge_task = asyncio.create_task(_purge_ticker())
    background_tasks_refs["purge_ticker"] = purge_task

    logger.info("Startup complete. Engines ready (bypass=%s).", bypass)

    yield

    # --- SHUTDOWN --------------------------------------------------------
    logger.info("CloudSentinel-XAI shutting down...")
    for name, task in background_tasks_refs.items():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await engine.dispose()
    state_matrix.clear()
    background_tasks_refs.clear()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="CloudSentinel-XAI",
    version="2.0.0",
    lifespan=lifespan,
)

# Expose the REAL pipeline metrics for Prometheus to scrape at /metrics.
# These are the same metric names grafana.json already queries, updated with
# genuine detector output from the ingest pipeline (not simulated values).
from prometheus_client import make_asgi_app  # noqa: E402
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    return {
        "status": "ONLINE",
        "engine": "cloud sentinel",
        # NOTE: this used to check state_matrix.get("iso_forest")/"xgboost",
        # keys nothing sets (see _load_ml_artifacts above, which sets
        # "iso_forest_bundle"/"xgboost_bundle") -- so /health always reported
        # bypass=True even with trained models loaded and scoring correctly.
        "ml_bypass": not (state_matrix.get("iso_forest_bundle") and state_matrix.get("xgboost_bundle")),
        "known_entities": (
            state_matrix["graph_engine"].get_entity_stats()["known_entities"]
            if "graph_engine" in state_matrix else 0
        ),
    }


# Attach routers
from app.routers import ingest  # noqa: E402  (import after app/state defined)
app.include_router(ingest.router, prefix="/api/v1")