"""
DEMO 6 — Persistence Service (SQLite + write-coalescing flusher)
Shows: risk states buffered in memory get atomically flushed to SQLite, and
the tables (entity_risk_states, xai_alerts) are created and written correctly.
Run:  PYTHONPATH=. python demo/demo_6_persistence.py
"""
import asyncio, os, sqlite3, tempfile
from demo._util import banner, step, show, check, done

banner("DEMO 6 — PERSISTENCE  (write-coalescing flusher -> SQLite)")

# isolate a temp DB for the demo
os.environ.setdefault("DEMO", "1")
tmp = tempfile.mkdtemp()
os.chdir(tmp)

from app.core.database import engine, Base, AsyncSessionLocal  # noqa
import app.services.db_flusher as flusher

async def run():
    step("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    check("tables created", os.path.exists("cloud_sentinel.db"))

    step("Buffering 3 entity risk states (as the pipeline would)...")
    for i, eid in enumerate(["actorA", "actorB", "actorC"]):
        flusher.pending_risk_updates[eid] = {
            "risk_intensity": 1.0 + i, "scaled_score": 0.5 + i*0.1,
            "cloud_span_count": i + 1, "is_critical": i == 2}
    show("buffer size before flush", len(flusher.pending_risk_updates))
    check("buffer holds 3 updates", len(flusher.pending_risk_updates) == 3)

    step("Running ONE flush cycle (normally the 3s background ticker)...")
    task = asyncio.create_task(flusher.risk_flush_ticker(flush_interval_seconds=1))
    await asyncio.sleep(1.6)
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    show("buffer size after flush", len(flusher.pending_risk_updates))
    check("buffer drained (atomic swap, no lost updates)", len(flusher.pending_risk_updates) == 0)

    step("Reading rows back from SQLite...")
    con = sqlite3.connect("cloud_sentinel.db")
    rows = con.execute("SELECT entity_id, scaled_score, cloud_span_count, is_critical "
                       "FROM entity_risk_states ORDER BY entity_id").fetchall()
    for r in rows: show("persisted row", r)
    con.close()
    check("all 3 risk states persisted", len(rows) == 3)
    await engine.dispose()

asyncio.run(run())
done("DEMO 6: Persistence Service")