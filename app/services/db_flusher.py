# app/services/db_flusher.py
"""
Write-coalescing risk-state flusher. The ingest pipeline writes the latest
Hawkes risk state per entity into an in-memory buffer; this ticker drains
the buffer into the configured database (SQLite by default, Postgres in
production -- see app/core/database.py's DATABASE_URL) every few seconds,
keeping DB writes off the hot path.

`INSERT ... ON CONFLICT DO UPDATE` ("upsert") is dialect-specific in
SQLAlchemy: `sqlalchemy.dialects.sqlite.insert` and
`sqlalchemy.dialects.postgresql.insert` are different constructs, even
though both expose the same `.on_conflict_do_update(...)` method. Previously
this module hardcoded the sqlite one, so switching DATABASE_URL to Postgres
would have raised at the first flush. `_upsert` below picks the right one
from the engine's actual dialect, so this file needs no further changes to
support either backend.
"""

import asyncio
import logging
from app.core.database import AsyncSessionLocal, EntityRiskState, engine

logger = logging.getLogger(__name__)

if engine.dialect.name == "postgresql":
    from sqlalchemy.dialects.postgresql import insert as _upsert
else:
    # Also covers sqlite (the default) and is a reasonable fallback for any
    # other dialect that shares the SQLite insert-construct's shape.
    from sqlalchemy.dialects.sqlite import insert as _upsert

# Global write-coalescing buffer: entity_id -> latest risk state dict.
pending_risk_updates: dict = {}


async def risk_flush_ticker(flush_interval_seconds: int = 3):
    global pending_risk_updates
    logger.info("Starting async DB flush ticker (%ss interval).", flush_interval_seconds)
    try:
        while True:
            await asyncio.sleep(flush_interval_seconds)

            if not pending_risk_updates:
                continue

            # Atomic swap: rebind the module-global to a fresh dict and take
            # the old one. Any writes that arrive mid-flush land in the new
            # dict and are picked up next tick — nothing is dropped (the
            # previous copy()+clear() had a lost-update race in between).
            updates_to_flush = pending_risk_updates
            pending_risk_updates = {}

            async with AsyncSessionLocal() as session:
                try:
                    for entity_id, state in updates_to_flush.items():
                        base = _upsert(EntityRiskState).values(
                            entity_id=entity_id,
                            current_risk_intensity=state["risk_intensity"],
                            scaled_score=state["scaled_score"],
                            cloud_span_count=state["cloud_span_count"],
                            is_critical=state["is_critical"],
                        )
                        stmt = base.on_conflict_do_update(
                            index_elements=["entity_id"],
                            set_=dict(
                                current_risk_intensity=base.excluded.current_risk_intensity,
                                scaled_score=base.excluded.scaled_score,
                                cloud_span_count=base.excluded.cloud_span_count,
                                is_critical=base.excluded.is_critical,
                            ),
                        )
                        await session.execute(stmt)
                    await session.commit()
                    logger.debug("Flushed %d risk states to the database.", len(updates_to_flush))
                except Exception as db_err:
                    logger.error("DB flush failed: %s", db_err)
                    await session.rollback()
                    # Re-queue failed updates so they aren't lost (best effort).
                    for eid, st in updates_to_flush.items():
                        pending_risk_updates.setdefault(eid, st)
    except asyncio.CancelledError:
        logger.info("DB flush ticker terminated.")