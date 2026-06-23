import time
from prometheus_client import start_http_server, Gauge, Counter

# Define your custom research metrics
USER_TRUST_GAUGE = Gauge(
    'security_user_trust_score', 
    'Current real-time trust score of a cloud identity (0-100)',
    ['user_identity', 'cloud_provider']
)

THREAT_COUNTER = Counter(
    'security_pipeline_alerts_total',
    'Cumulative count of malicious events intercepted by parallel engines',
    ['cloud_provider', 'engine_source']
)

def simulate_pipeline_processing():
    # Initialize a mock user for testing the endpoint
    USER_TRUST_GAUGE.labels(user_identity="dev-user-01", cloud_provider="AWS").set(100.0)
    
    print("Security Metrics Server running on http://localhost:8000/metrics")
    
    # Mock loop simulating incoming threats dynamically modifying state
    current_score = 100.0
    while True:
        time.sleep(5)
        # Simulate a risk engine trigger dropping a user's trust score
        if current_score > 30:
            current_score -= 5.5
            USER_TRUST_GAUGE.labels(user_identity="dev-user-01", cloud_provider="AWS").set(current_score)
            THREAT_COUNTER.labels(cloud_provider="AWS", engine_source="XGBoost").inc()
            print(f"⚠️ Simulated Threat Intercepted. Trust score decayed to: {current_score}")

if __name__ == "__main__":
    # Start the local Prometheus metric scrapable HTTP server
    start_http_server(8000)
    simulate_pipeline_processing()