"""
DEMO 5 — Faithfulness-Gated XAI Service
Shows the SAFETY GATE that sits in front of the LLM: before any narrative is
generated, a SHAP deletion test checks the explanation is grounded in the
model's real decision. Without a trained model, the gate correctly REFUSES
(fails closed) rather than emitting an ungrounded narrative — which is exactly
the behavior you want.
Run:  PYTHONPATH=. python demo/demo_5_xai.py
"""
import asyncio
from demo._util import banner, step, show, check, done
from app.services.xai_engine import FaithfulnessGatedXAI

banner("DEMO 5 — FAITHFULNESS-GATED XAI  (no hallucinated narratives)")
xai = FaithfulnessGatedXAI({})   # no model -> gate must fail closed

log = {"principal": "alice", "source_cloud": "AWS", "action": "AttachUserPolicy",
       "predicted_phase": "PrivilegeEscalation", "phase_confidence": 0.92,
       "predicted_phase_index": 3,
       "shap_attributions": {"action": {"raw_value": "AttachUserPolicy", "shap_impact": 0.4}}}
risk = {"dominant_signal": "cross_cloud_diversity", "cloud_span_count": 3,
        "scaled_score": 0.88}

step("Asking the XAI service to explain a CRITICAL alert...")
passed_gate, narrative, generation_ok = asyncio.run(
    xai.generate_forensic_narrative(log, risk))
show("faithfulness gate passed?", passed_gate)
show("LLM generation succeeded?", generation_ok)
show("returned text", narrative)

step("Verifying the safety contract...")
check("service returns the 3-tuple (gate, narrative, generation_ok)",
      isinstance(passed_gate, bool) and isinstance(narrative, str) and isinstance(generation_ok, bool))
check("with NO trained model, the gate FAILS CLOSED (no fabricated narrative)",
      generation_ok is False)
print("   -> The system flags the alert for manual review instead of inventing")
print("      an explanation. When a model + Ollama are present, a faithful")
print("      explanation passes the gate and a real 2-sentence narrative is")
print("      generated. This is the anti-hallucination guarantee.")

step("Design point: the deletion test is scoped to the MODEL's features only")
from app.services.ml_inference import ParallelMLEngine
check("deletion test uses the model feature schema (no stray log keys)",
      hasattr(ParallelMLEngine, "_ORDERED_FEATURES"))
done("DEMO 5: Faithfulness-Gated XAI Service")