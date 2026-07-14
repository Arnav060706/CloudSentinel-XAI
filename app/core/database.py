# app/core/database.py
import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import event

logger = logging.getLogger(__name__)

# The SQLAlchemy Abstraction Layer
DATABASE_URL = "sqlite+aiosqlite:///./cloud_sentinel.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

# Enable Write-Ahead Logging (WAL) for SQLite concurrent reads
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

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