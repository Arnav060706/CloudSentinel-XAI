# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import ingest
from app.services.graph_engine import MultiCloudGraphEngine
import xgboost as xgb
import pickle

# Establish Global Cache Handlers
state_matrix = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP TIMELINE ---
    # Load resource-intensive binary models directly into memory once
    with open("models/isolation_forest.pkl", "rb") as f:
        state_matrix["iso_forest"] = pickle.load(f)
        
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model("models/xgboost_model.json")
    state_matrix["xgboost"] = xgb_clf
    
    # Instantiate our stateful O(1) in-memory directed graph engine
    state_matrix["graph_engine"] = MultiCloudGraphEngine()
    
    yield
    # --- SHUTDOWN TIMELINE ---
    state_matrix.clear()

app = FastAPI(
    title="CloudSentinel-XAI v2 Backend Core", 
    version="2.0.0",
    lifespan=lifespan
)

# Attach Routers
app.include_router(ingest.router, prefix="/api/v1")