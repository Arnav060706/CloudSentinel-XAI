import time
import random
from prometheus_client import start_http_server, Gauge, Counter
from aggregator.xai_triage import generate_soc_narrative

# 1. Stateful Trust Layer Metric (Layer 4)
USER_TRUST_GAUGE = Gauge(
    'security_user_trust_score', 
    'Current real-time trust score of a cloud identity (0-100)',
    ['user_identity', 'cloud_provider']
)

# 2. Parallel Analytics Engine Metrics (Layer 3)
ENGINE_RISK_PROBABILITY = Gauge(
    'security_engine_risk_probability',
    'Real-time risk score or probability output by independent analytical engines',
    ['engine_name', 'cloud_provider']
)

PIPELINE_THROUGHPUT = Counter(
    'security_pipeline_processed_events_total',
    'Total number of multi-cloud log events processed by the core normalization pipeline'
)

NARRATIVE_EXPORTER = Gauge(
    'security_triage_narrative_info',
    'Current active natural language triage narrative from local LLM',
    ['alert_id','user_identity', 'cloud_provider', 'event_action', 'risk_score', 'narrative_text']
)

def run_security_pipeline():
    print("Multi-Engine Security Metrics Server running on http://localhost:8000/metrics")
    start_http_server(8000)
    
    current_trust = 100.0
    user_id = "dev-user-01"
    
    while True:
        time.sleep(5) # Simulating your 5-second Grafana refresh window
        
        # Increment global pipeline throughput metric
        incoming_logs_count = random.randint(10, 50)
        PIPELINE_THROUGHPUT.inc(incoming_logs_count)
        
        # Simulate a random threat scenario shifting engine parameters
        if random.random() < 0.3: # 30% chance of an anomalous event trigger
            # 1. Identity Engine (Isolation Forest) finds an anomaly
            iso_forest_risk = round(random.uniform(0.75, 0.95), 2)
            # 2. Network Engine (XGBoost) evaluates traffic as mostly benign
            xgboost_risk = round(random.uniform(0.10, 0.30), 2)
            # 3. Static Rules remain untriggered (0)
            rule_trigger = 0.0
            
            # Apply your Stateful Trust Score Decay Formula:
            # Deduction = (0.6 * IsolationForest) + (0.4 * XGBoost)
            deduction = (15 * iso_forest_risk) + (10 * xgboost_risk)
            current_trust = max(0.0, current_trust - deduction)
            
            print(f"Anomaly Detected! IsoForest: {iso_forest_risk}, XGBoost: {xgboost_risk} | Trust Decayed to: {current_trust:.2f}")
            #Trigger the local LLM if trust drops below 60
            if current_trust < 50:
                print("\n[!] Trust score in critical boundary. Querying local XAI LLM Aggregator...")
                #packaging current runtime metrics into a mock shap vector
                current_shap = {
                    "isolation_forest_anamoly_probability": iso_forest_risk,
                    "xgboost_threat_confidnece": xgboost_risk
                }
                narrative = generate_soc_narrative(
                    cloud_provider="AWS",
                    event_name="AttachUserPolicy",
                    risk_score=int((iso_forest_risk*100)),
                    shap_features=current_shap
                )
                #print(f"LLM Triage Report:\n{narrative}")
                clean_narrative = narrative.replace('"', "'").replace('\n', ' ')
                # Generate a unique short string ID for this specific clock tick
                unique_alert_id = f"ALERT-{int(time.time())}"
                NARRATIVE_EXPORTER.clear()
                NARRATIVE_EXPORTER.labels(
                    alert_id=unique_alert_id,
                    user_identity="dev-user-01",
                    cloud_provider="AWS",
                    event_action="UpdateAssumeRolePolicy",
                    risk_score="87.5",
                    narrative_text=clean_narrative
                ).set(1)
        else:
            # Benign cycle: Engines report low risk, trust slowly recovers (+1)
            iso_forest_risk = round(random.uniform(0.01, 0.15), 2)
            xgboost_risk = round(random.uniform(0.01, 0.12), 2)
            rule_trigger = 0.0
            current_trust = min(100.0, current_trust + 1.0)
        
        # Update Prometheus fields with calculated telemetry variables
        USER_TRUST_GAUGE.labels(user_identity=user_id, cloud_provider="AWS").set(current_trust)
        ENGINE_RISK_PROBABILITY.labels(engine_name="Isolation_Forest", cloud_provider="AWS").set(iso_forest_risk)
        ENGINE_RISK_PROBABILITY.labels(engine_name="XGBoost", cloud_provider="AWS").set(xgboost_risk)
        ENGINE_RISK_PROBABILITY.labels(engine_name="Rule_Engine", cloud_provider="AWS").set(rule_trigger)

if __name__ == "__main__":
    run_security_pipeline()