"""
DEMO 4 — Cross-Cloud Risk Engine
Shows the three rules in action:
  1. anomaly adds risk   2. old risk decays   3. cross-cloud multiplies risk
Plus the DevOps false-positive fix (baseline suppresses legitimate multi-cloud).
Run:  PYTHONPATH=. python demo/demo_4_risk.py
"""
import time, datetime
from demo._util import banner, step, show, check, done
from app.services.risk_engine import RiskEngine

banner("DEMO 5 — CROSS-CLOUD RISK ENGINE  (decayed risk + cloud multiplier)")
eng = RiskEngine()
now = time.time()
iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()

def ev(cloud, anomaly, age_s, principal="alice", ptype="IAMUser", ua="Chrome"):
    return {"source_cloud": cloud, "anomaly_score": anomaly,
            "timestamp": iso(now - age_s), "principal": principal,
            "principal_type": ptype, "ua_family": ua}

step("RULE 3 — one anomalous event, SINGLE cloud vs THREE clouds")
single = eng.calculate_intensity([ev("AWS", 0.9, 2)], lifetime_clouds=["AWS"],
                                 eval_time=now, entity_id="u1")
multi = eng.calculate_intensity([ev("AWS", 0.9, 2)], lifetime_clouds=["AWS", "AZURE", "GCP"],
                                eval_time=now, entity_id="u2")
show("single-cloud  -> multiplier", single["diversity_multiplier"])
show("single-cloud  -> risk", single["risk_intensity"])
show("three-cloud   -> multiplier", multi["diversity_multiplier"])
show("three-cloud   -> risk", multi["risk_intensity"])
check("crossing clouds increases the multiplier", multi["diversity_multiplier"] > single["diversity_multiplier"])
check("crossing clouds increases the risk", multi["risk_intensity"] > single["risk_intensity"])

step("RULE 2 — the SAME event, fresh (2s old) vs stale (300s old)")
fresh = eng.calculate_intensity([ev("AWS", 0.9, 2)], lifetime_clouds=["AWS"], eval_time=now, entity_id="a")
stale = eng.calculate_intensity([ev("AWS", 0.9, 300)], lifetime_clouds=["AWS"], eval_time=now, entity_id="b")
show("fresh event risk", fresh["risk_intensity"])
show("stale event risk", stale["risk_intensity"])
check("old risk decays toward baseline", stale["risk_intensity"] < fresh["risk_intensity"])

step("DevOps FALSE-POSITIVE FIX — baseline suppresses legitimate multi-cloud")
hot = eng.calculate_intensity([ev("AWS", 0.9, 2)], lifetime_clouds=["AWS", "AZURE", "GCP"],
                              eval_time=now, entity_id="dev-unbaselined")
eng.record_baseline("dev-known", ["AWS", "AZURE", "GCP"])   # this identity is NORMALLY multi-cloud
calm = eng.calculate_intensity([ev("AWS", 0.9, 2)], lifetime_clouds=["AWS", "AZURE", "GCP"],
                               eval_time=now, entity_id="dev-known")
show("unbaselined 3-cloud actor -> multiplier", hot["diversity_multiplier"])
show("baselined DevOps actor    -> multiplier", calm["diversity_multiplier"])
check("known-multi-cloud DevOps identity is NOT amplified", calm["diversity_multiplier"] == 1.0)
check("novel_cloud_span_count is 0 for the baselined actor", calm["novel_cloud_span_count"] == 0)

step("Automation vs human — automation gets a gentler multiplier")
human = eng.calculate_intensity([ev("AWS", 0.9, 2, ptype="IAMUser", ua="Chrome")],
                                lifetime_clouds=["AWS", "AZURE"], eval_time=now, entity_id="h")
autom = eng.calculate_intensity([ev("AWS", 0.9, 2, ptype="ServiceAccount", ua="boto3")],
                                lifetime_clouds=["AWS", "AZURE"], eval_time=now, entity_id="s")
show("human multiplier", human["diversity_multiplier"])
show("automation multiplier", autom["diversity_multiplier"])
check("automation is amplified less than a human for the same cross", autom["diversity_multiplier"] < human["diversity_multiplier"])
done("DEMO 5: Cross-Cloud Risk Engine")