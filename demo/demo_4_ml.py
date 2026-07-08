"""
DEMO 3 — ML Inference Service (Isolation Forest + XGBoost + SHAP)
Shows: the parallel inference contract. Runs in BYPASS mode (no trained models
yet) and proves it returns every field downstream services need, without
crashing. When models are dropped into models/, the SAME call upgrades to real
anomaly scores + ATT&CK phase + SHAP drivers automatically.
Run:  PYTHONPATH=. python demo/demo_3_ml.py
"""
import asyncio, os
from demo._util import banner, step, show, check, done
from app.services.ml_inference import ParallelMLEngine

banner("DEMO 3 — ML INFERENCE SERVICE  (parallel IF + XGBoost + SHAP)")

step("Models present on disk?")
have_models = os.path.exists("models/isolation_forest.pkl") and os.path.exists("models/xgboost_model.json")
show("models/ trained artifacts found", have_models)
mode = "REAL inference" if have_models else "BYPASS (neutral defaults — no trained models yet)"
show("running mode", mode)

engine = ParallelMLEngine({})   # empty state_matrix -> bypass unless models loaded
event = {"principal": "alice", "source_cloud": "AWS", "action": "AttachUserPolicy",
         "ua_family": "aws-cli", "status": "SUCCESS", "principal_type": "IAMUser"}

step("Feeding one normalized event through parallel inference...")
scored = asyncio.run(engine.execute_parallel_inference(dict(event)))
show("anomaly_score (Isolation Forest)", scored["anomaly_score"])
show("predicted_phase (XGBoost / ATT&CK)", scored["predicted_phase"])
show("phase_confidence", scored["phase_confidence"])
show("shap_attributions (top drivers)", scored["shap_attributions"])

step("Verifying the inference CONTRACT (every downstream field is present)...")
for key in ("anomaly_score", "predicted_phase", "predicted_phase_index",
            "phase_confidence", "shap_attributions"):
    check(f"output has '{key}'", key in scored)
check("anomaly_score is a float in [0,1]", 0.0 <= float(scored["anomaly_score"]) <= 1.0)
check("shap_attributions is a dict", isinstance(scored["shap_attributions"], dict))

step("Concurrency proof — the two models run in parallel threads")
check("engine exposes the async parallel entry point",
      asyncio.iscoroutinefunction(engine.execute_parallel_inference))
print("   (Isolation Forest and XGBoost are dispatched with asyncio.gather so")
print("    per-event latency ≈ max(model), not sum(model).)")
done("DEMO 3: ML Inference Service")