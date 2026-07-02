# app/services/db_flusher.py
import asyncio
import logging
from sqlalchemy.dialects.sqlite import insert
from app.core.database import AsyncSessionLocal, EntityRiskState

logger = logging.getLogger(__name__)

# Global Write-Coalescing Buffer
# Maps entity_id -> dict of latest Hawkes Risk State
pending_risk_updates = {}

async def risk_flush_ticker(flush_interval_seconds: int = 3):
    """
    Background asynchronous loop that drains the memory buffer into SQLite.
    Prevents database write-locks from blocking the main FastAPI event loop.
    """
    logger.info(f"Starting async database flush ticker ({flush_interval_seconds}s interval).")
    
    try:
        while True:
            await asyncio.sleep(flush_interval_seconds)
            
            if not pending_risk_updates:
                continue
                
            # Safely extract and clear the buffer
            updates_to_flush = pending_risk_updates.copy()
            pending_risk_updates.clear()
            
            async with AsyncSessionLocal() as session:
                try:
                    for entity_id, state in updates_to_flush.items():
                        # SQLite Upsert (Insert or Update if exists)
                        stmt = insert(EntityRiskState).values(
                            entity_id=entity_id,
                            current_risk_intensity=state["risk_intensity"],
                            scaled_score=state["scaled_score"],
                            cloud_span_count=state["cloud_span_count"],
                            is_critical=state["is_critical"]
                        )
                        
                        # If the entity is already in the database, update its scores
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['entity_id'],
                            set_=dict(
                                current_risk_intensity=stmt.excluded.current_risk_intensity,
                                scaled_score=stmt.excluded.scaled_score,
                                cloud_span_count=stmt.excluded.cloud_span_count,
                                is_critical=stmt.excluded.is_critical
                            )
                        )
                        await session.execute(stmt)
                        
                    await session.commit()
                    logger.debug(f"Flushed {len(updates_to_flush)} risk states to SQLite.")
                    
                except Exception as db_err:
                    logger.error(f"Database flush failed: {db_err}")
                    await session.rollback()
                    
    except asyncio.CancelledError:
        logger.info("Database flush ticker safely terminated.")