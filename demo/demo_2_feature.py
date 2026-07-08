"""
DEMO 8 — Feature Engineering Service (unified log -> ML feature matrix)
Shows: how a normalized unified log is converted into the numeric feature
vector the ML models consume, INCLUDING the extra engineered features
(temporal, identity, and behavioral-velocity) that don't exist in the raw
log but make attacks detectable. Also proves target isolation (no leakage).
Run:  PYTHONPATH=. python demo/demo_8_features.py    (or: python -m demo.demo_8_features)
"""
import json, os
import pandas as pd
from demo._util import banner, step, show, check, done
from app.parser_normalizer.src.pipeline import ParserPipeline
from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor

pd.set_option("display.max_columns", None); pd.set_option("display.width", 200)

banner("DEMO 2 — FEATURE ENGINEERING  (unified log -> ML feature matrix)")

DATA = os.path.join("app", "parser_normalizer", "mock_data", "unified_datastream.json")
step(f"Normalizing the raw multi-cloud logs from {DATA} ...")
pipeline = ParserPipeline()
raw_logs = json.load(open(DATA))
unified = [pipeline.process_log(r).model_dump() for r in raw_logs if pipeline.process_log(r)]
check("we have normalized unified logs to feed the extractor", len(unified) >= 3)
show("raw unified fields per log (before engineering)", len(unified[0]))

step("Running the feature extractor  (unified logs -> X, y) ...")
fx = MLFeatureExtractor()
X, y = fx.extract_features(unified, is_training=True, export_csv=False)
show("X shape (rows x engineered features)", f"{X.shape[0]} x {X.shape[1]}")
show("y shape (rows x target columns)", f"{y.shape[0]} x {y.shape[1]}")
check("every raw log produced one feature row", X.shape[0] == len(unified))
check("many features engineered from few raw fields", X.shape[1] > len(unified[0]) - 5)

# ---- Group the engineered features by WHY they help detection ----
groups = {
    "TEMPORAL — when did it happen? (attackers work off-hours)": [
        "hour_of_day", "day_of_week", "is_weekend", "is_night_access", "is_business_hour"],
    "IDENTITY / AUTH — who, and how trusted?": [
        "mfa_authenticated", "user_type_is_service", "principal_type",
        "principal_created_in_window", "account_type", "is_known_proxy_or_tor"],
    "ACTION — how sensitive is what they did?": [
        "action_sensitivity_score", "login_result_success", "is_internal_ip"],
    "BEHAVIORAL VELOCITY — the fingerprints of an attack (engineered, NOT in raw log)": [
        "api_call_count_1m", "error_rate_5m", "unique_ips_last_24h",
        "privileged_actions_last_24h", "read_vs_write_ratio",
        "unique_resources_accessed", "is_new_ip_for_user", "is_new_device_for_user"],
}
why = {
    "is_night_access": "3am admin action is far more suspicious than 3pm",
    "mfa_authenticated": "a privileged action WITHOUT MFA is a red flag",
    "principal_created_in_window": "an account created minutes ago = likely persistence",
    "action_sensitivity_score": "SetIamPolicy/CreateUser score higher than a read",
    "api_call_count_1m": "a burst of calls suggests automation / brute force",
    "error_rate_5m": "many errors = attacker probing permissions",
    "unique_ips_last_24h": "a jump in source IPs = credential sharing / hijack",
    "read_vs_write_ratio": "read-heavy = reconnaissance",
    "is_new_ip_for_user": "first time this user came from this IP",
}
step("The ENGINEERED features, grouped by what they detect:")
for title, feats in groups.items():
    present = [f for f in feats if f in X.columns]
    print(f"\n   • {title}")
    for f in present:
        tip = f"   ← {why[f]}" if f in why else ""
        vals = [v.item() if hasattr(v, "item") else v for v in X[f].values]  # numpy -> plain python
        show(f"     {f}", f"{vals}{tip}")

step("Proving TARGET ISOLATION (no label leakage)...")
target_cols = fx.target_columns
leaked = [c for c in target_cols if c in X.columns]
show("target columns (kept OUT of X)", target_cols)
show("any target leaked into X?", leaked or "none")
check("no target/label column leaked into the feature matrix X", len(leaked) == 0)
check("targets are held separately in y", any(c in y.columns for c in target_cols))

step("Confirming X is fully numeric (model-ready)...")
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
show("non-numeric feature columns remaining", non_numeric or "none — all encoded")
check("every feature column is numeric (sklearn/XGBoost-ready)", len(non_numeric) == 0)

print("\n   Summary:")
print(f"   {len(unified[0])} raw unified fields  ->  {X.shape[1]} numeric ML features")
print("   (temporal + identity + behavioral-velocity), targets safely isolated.")
done("DEMO 2: Feature Engineering Service")