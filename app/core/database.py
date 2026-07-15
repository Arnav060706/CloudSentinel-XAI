# app/core/database.py
"""
Database engine/session setup.

Backend is now chosen by the DATABASE_URL environment variable instead of
being hardcoded to a single SQLite file. This is what lets the exact same
codebase run against SQLite for local dev/tests (the default, zero-setup)
and against PostgreSQL in production, with no code changes on either side
-- just set DATABASE_URL and install the matching async driver:

    # default (unchanged): local SQLite file, good for dev/tests
    unset DATABASE_URL
    # or: DATABASE_URL=sqlite+aiosqlite:///./cloud_sentinel.db

    # PostgreSQL (recommended for anything beyond a single dev box):
    DATABASE_URL=postgresql+asyncpg://user:password@host:5432/cloudsentinel
    # requires: pip install asyncpg

Why this matters for CloudSentinel specifically: db_flusher.py, main.py's
startup, and the ingest pipeline all write concurrently under real load
(write-coalesced risk-state updates every few seconds, plus XAIAlert and
LLMBenchmark inserts on every critical alert). SQLite's single-writer model
and file-level locking is fine for a single dev/demo process, but becomes a
bottleneck and an operational liability (one file, no replication, no
concurrent-writer scaling, easy to corrupt on a hard crash) the moment this
runs as more than one process or needs real uptime guarantees. Postgres
removes all of that with the same SQLAlchemy models and the same upsert
call sites (see db_flusher.py's dialect-aware insert()).
"""
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import event

logger = logging.getLogger(__name__)

# The SQLAlchemy Abstraction Layer. Defaults to the original local SQLite
# file so existing dev setups / `uvicorn app.main:app` / the test suite need
# zero configuration; set DATABASE_URL to point at Postgres in production.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./cloud_sentinel.db")
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

_engine_kwargs = {"echo": False}
if _IS_SQLITE:
    # check_same_thread is a SQLite/DBAPI-specific connect arg; passing it to
    # asyncpg's connect() raises TypeError, so it must only apply here.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)

if _IS_SQLITE:
    # Enable Write-Ahead Logging (WAL) for SQLite concurrent reads. These
    # PRAGMAs are meaningless (and the hook itself is never fired) on
    # Postgres, which has its own, far more capable MVCC/WAL implementation
    # out of the box -- nothing to configure here for that path.
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
else:
    logger.info("Using non-SQLite database backend: %s", engine.url.get_backend_name())

AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base = declarative_base()

# --- Table Definitions ---

class EntityRiskState(Base):
    """Stores the active Hawkes process risk scores for Grafana dashboards."""
    __tablename__ = "entity_risk_states"
    
    entity_id = Column(String, primary_key=True, index=True)
    current_risk_intensity = Column(Float, default=0.0)
    scaled_score = Column(Float, default=0.0)
    cloud_span_count = Column(Integer, default=1)
    is_critical = Column(Boolean, default=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class XAIAlert(Base):
    """Permanent ledger for Llama 3.2 generated forensic narratives."""
    __tablename__ = "xai_alerts"
    
    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String, ForeignKey("entity_risk_states.entity_id"), nullable=False)
    predicted_phase = Column(String, nullable=False)
    dominant_shap_signal = Column(String, nullable=False)
    llama_narrative = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class LLMBenchmark(Base):
    """
    Latency/quality telemetry for the XAI narrative LLM call. Written
    fire-and-forget from FaithfulnessGatedXAI so it never adds latency to
    the critical path. Query this table to build the
    model / avg-latency / prompt-size / cache-hit-rate comparison table
    used to justify prompt and model choices.
    """
    __tablename__ = "llm_benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    predicted_phase = Column(String, nullable=False)
    cache_hit = Column(Boolean, default=False)
    prompt_chars = Column(Integer, default=0)
    completion_chars = Column(Integer, default=0)
    elapsed_seconds = Column(Float, default=0.0)
    succeeded = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())