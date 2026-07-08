"""
DEMO 1 — Parser & Normalizer (Service: multi-cloud log ingestion)
Shows: the THREE different raw cloud log formats in our mock datastream
(AWS / Azure / GCP) each get auto-detected and normalized into ONE unified
schema. Reads the SAME raw file the end-to-end pipeline uses, so the demo is
consistent with the full run.
Run:  PYTHONPATH=. python demo/demo_1_parser.py
"""
import json, os
from demo._util import banner, step, show, check, done
from app.parser_normalizer.src.pipeline import ParserPipeline

banner("DEMO 1 — PARSER & NORMALIZER  (3 raw cloud formats -> 1 unified schema)")
pipeline = ParserPipeline()

DATA = os.path.join("app", "parser_normalizer", "mock_data", "unified_datastream.json")
step(f"Loading RAW multi-cloud logs from {DATA} ...")
raw_logs = json.load(open(DATA))
check("loaded a list of raw logs", isinstance(raw_logs, list) and len(raw_logs) >= 3)
show("number of raw records", len(raw_logs))

# Quick proof these really are different provider-native formats
step("Confirming the records are genuinely DIFFERENT provider formats...")
for i, raw in enumerate(raw_logs):
    sig = "AWS" if "userIdentity" in raw else "AZURE" if "callerIpAddress" in raw else "GCP" if "protoPayload" in raw else "UNKNOWN"
    show(f"record {i} native format (by signature keys)", sig)
check("the raw records are NOT already normalized (different key sets each)",
      len({frozenset(r.keys()) for r in raw_logs}) == len(raw_logs))

unified_keysets = []
for i, raw in enumerate(raw_logs):
    step(f"Normalizing raw record {i} (auto-detecting provider)...")
    unified = pipeline.process_log(raw)
    check(f"record {i} parsed successfully (not None)", unified is not None)
    d = unified.model_dump()
    unified_keysets.append(frozenset(d.keys()))
    show("detected source_cloud", d.get("source_cloud"))
    show("normalized action", d.get("action"))
    show("normalized user_id", d.get("user_id"))
    show("normalized source_ip", str(d.get("source_ip")))
    check("output conforms to unified schema (timestamp+action+user_id+source_cloud)",
          all(k in d for k in ("timestamp", "action", "user_id", "source_cloud")))
    # Full unified record so the panel sees the whole normalized schema.
    # (We hide the bulky nested raw_log from the printout for readability —
    #  it's still carried in the object.)
    display = {k: str(v) for k, v in d.items() if k != "raw_log"}
    show(f"UNIFIED LOG (record {i}, raw_log hidden for readability)", display)

step("Confirming ALL formats collapsed to the SAME unified structure...")
show("distinct unified key-sets after normalization", len(set(unified_keysets)))
check("all providers now share ONE identical schema", len(set(unified_keysets)) == 1)
print("   -> Different raw in (AWS/Azure/GCP), one schema out. This is the")
print("      normalization that makes all downstream logic cloud-agnostic.")
done("DEMO 1: Parser & Normalizer")