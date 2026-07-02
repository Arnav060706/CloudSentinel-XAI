# app/main.py
#from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import ingest
#from app.services.graph_engine import MultiCloudGraphEngine
#import xgboost as xgb
#import pickle

# Establish Global Cache Handlers
# state_matrix = {}

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # --- STARTUP TIMELINE ---
#     # 1. Initialize SQLite Database Tables via SQLAlchemy
#        async with engine.begin() as conn:
#            await conn.run_sync(Base.metadata.create_all)
#     # Load resource-intensive binary models directly into memory once
#     with open("models/isolation_forest.pkl", "rb") as f:
#         state_matrix["iso_forest"] = pickle.load(f)
        
#     xgb_clf = xgb.XGBClassifier()
#     xgb_clf.load_model("models/xgboost_model.json")
#     state_matrix["xgboost"] = xgb_clf
    
#     # Instantiate our stateful O(1) in-memory directed graph engine
#     state_matrix["graph_engine"] = MultiCloudGraphEngine()

#     # 3. Spin up the Background Flush Ticker
#     flush_task = asyncio.create_task(risk_flush_ticker(flush_interval_seconds=3))
#     background_tasks_refs["flush_ticker"] = flush_task
    
#     yield
#     # --- SHUTDOWN TIMELINE ---
      # 1. Cancel the flush ticker gracefully
#      if "flush_ticker" in background_tasks_refs:
#        background_tasks_refs["flush_ticker"].cancel()
        
      # 2. Close database connections
#      await engine.dispose()
#     state_matrix.clear()

app = FastAPI(
    title="CloudSentinel-XAI", 
    version="2.0.0",
    #lifespan=lifespan
)

@app.get("/health")
async def trail():
    return {
        "status": "ONLINE",
        "engine": "cloud sentinel"
    }

# Attach Routers
app.include_router(ingest.router, prefix="/api/v1")