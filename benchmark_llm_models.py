"""
benchmark_llm_models.py
========================
Compares candidate LLM models for the XAI narrative layer on latency and
prompt/response size, using the EXACT prompt-building logic from
app/services/xai_engine.py::FaithfulnessGatedXAI._build_prompt — so results
are representative of what production actually sends, not a hand-typed
approximation.

Does NOT require a trained XGBoost model or a passing faithfulness gate:
it calls the LLM directly with representative synthetic SHAP payloads, so
you can run this benchmark independently of your teammate's model training
work.

Usage
-----
    ollama pull llama3.2
    ollama pull qwen3:4b
    ollama pull qwen3:8b

    python benchmark_llm_models.py --models llama3.2 qwen3:4b qwen3:8b

Prints a comparison table and writes results to benchmark_results.csv.
"""

import argparse
import asyncio
import csv
import time
from pathlib import Path

from app.services.xai_engine import FaithfulnessGatedXAI

# A handful of representative alert shapes spanning different ATT&CK
# phases, pulled from the same signal vocabulary your dataset generator
# and ml_inference.py actually produce.
SAMPLE_ALERTS = [
    {
        "log_data": {
            "principal": "alice.chen@corp.com",
            "source_cloud": "AZURE",
            "predicted_phase": "Credential Access",
            "phase_confidence": 0.968,
            "shap_attributions": {
                "is_known_proxy_or_tor": {"raw_value": True, "shap_impact": 0.32},
                "geo_country": {"raw_value": "RO", "shap_impact": 0.21},
                "action": {"raw_value": "ConsoleLogin", "shap_impact": 0.18},
            },
        },
        "risk_state": {"dominant_signal": "impossible_travel", "cloud_span_count": 2},
    },
    {
        "log_data": {
            "principal": "svc-deploy-bot",
            "source_cloud": "AWS",
            "predicted_phase": "Privilege Escalation",
            "phase_confidence": 0.91,
            "shap_attributions": {
                "action": {"raw_value": "AttachRolePolicy", "shap_impact": 0.29},
                "principal_created_in_window": {"raw_value": True, "shap_impact": 0.24},
                "principal_type": {"raw_value": "IAMUser", "shap_impact": 0.15},
            },
        },
        "risk_state": {"dominant_signal": "privilege_escalation_burst", "cloud_span_count": 1},
    },
    {
        "log_data": {
            "principal": "bob.smith@corp.com",
            "source_cloud": "GCP",
            "predicted_phase": "Exfiltration",
            "phase_confidence": 0.87,
            "shap_attributions": {
                "action": {"raw_value": "storage.objects.list", "shap_impact": 0.27},
                "ua_family": {"raw_value": "python-requests", "shap_impact": 0.19},
            },
        },
        "risk_state": {"dominant_signal": "cross_cloud_data_staging", "cloud_span_count": 3},
    },
]


async def run_one(engine: FaithfulnessGatedXAI, model_name: str, sample: dict) -> dict:
    engine.model_name = model_name
    system_instruction, user_prompt = engine._build_prompt(
        sample["log_data"],
        sample["risk_state"],
        sample["log_data"]["shap_attributions"],
        sample["log_data"]["predicted_phase"],
        sample["log_data"]["phase_confidence"],
    )

    start = time.perf_counter()
    succeeded = True
    completion = ""
    try:
        response = await engine._chat_with_retry(system_instruction, user_prompt)
        completion = response.get("message", {}).get("content", "").strip()
    except Exception as e:
        succeeded = False
        completion = f"[ERROR: {e}]"
    elapsed = time.perf_counter() - start

    return {
        "model": model_name,
        "phase": sample["log_data"]["predicted_phase"],
        "prompt_chars": len(user_prompt),
        "completion_chars": len(completion),
        "elapsed_s": round(elapsed, 3),
        "succeeded": succeeded,
        "completion_preview": completion[:120],
    }


async def main(models: list[str]):
    engine = FaithfulnessGatedXAI(state_matrix={})  # no trained model needed for this benchmark
    if engine.llm_client is None:
        print("ollama is not available in this environment — install/enable it first.")
        return

    results = []
    for model_name in models:
        print(f"\n=== {model_name} ===")
        for sample in SAMPLE_ALERTS:
            row = await run_one(engine, model_name, sample)
            results.append(row)
            print(f"  {row['phase']:<24} {row['elapsed_s']:>6.3f}s  "
                  f"prompt={row['prompt_chars']}c  completion={row['completion_chars']}c  "
                  f"ok={row['succeeded']}")

    out_path = Path("benchmark_results.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {out_path.resolve()}")

    print("\n--- Summary (avg latency per model) ---")
    for model_name in models:
        model_rows = [r for r in results if r["model"] == model_name and r["succeeded"]]
        if not model_rows:
            print(f"  {model_name:<15} no successful calls")
            continue
        avg_latency = sum(r["elapsed_s"] for r in model_rows) / len(model_rows)
        avg_prompt = sum(r["prompt_chars"] for r in model_rows) / len(model_rows)
        print(f"  {model_name:<15} avg_latency={avg_latency:.3f}s  avg_prompt_chars={avg_prompt:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama3.2"],
                         help="Ollama model tags to compare, e.g. llama3.2 qwen3:4b qwen3:8b")
    args = parser.parse_args()
    asyncio.run(main(args.models))