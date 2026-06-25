import json
from src.pipeline import ParserPipeline

def detect_cloud_source(raw_log: dict) -> str:
    """Heuristic to auto-detect the cloud provider from raw log structure."""
    if "awsRegion" in raw_log or ("eventSource" in raw_log and "amazonaws.com" in raw_log["eventSource"]):
        return "AWS"
    if "tenantId" in raw_log or "resourceProviderName" in raw_log:
        return "AZURE"
    return "UNKNOWN"

def main():
    pipeline = ParserPipeline()

    print("=" * 60)
    print("STARTING ZERO-TRUST LOG INGESTION & NORMALIZATION PIPELINE")
    print("=" * 60)

    try:
        with open("mock_data/sample_logs_aws.json", "r") as f:
            mock_entries = json.load(f)
    except FileNotFoundError:
        print("[ERROR] sample_logs.json not found.")
        return

    for idx, entry in enumerate(mock_entries, 1):
        # 1. Auto-detect the source based on the raw log keys
        cloud_source = detect_cloud_source(entry)
        payload = entry
        
        print(f"\n[Processing Log #{idx}] Detected Source: {cloud_source} | Event: {payload.get('eventName', 'N/A')}")
        
        normalized_record = pipeline.process_log(cloud_source, payload)
        
        if normalized_record:
            print("Status: SUCCESS ✅")
            print(f"Severity: {normalized_record.severity} | Threat Category: {payload.get('ml_labels', {}).get('threat_category', 'None')}")
            # print(normalized_record.model_dump_json(indent=2)) # Uncomment to see full JSON
        else:
            print("Status: FAILED ❌")

    print("\n" + "=" * 60)
    print(f"PIPELINE SUMMARY - FAILED LOGS STORE ({len(pipeline.failed_logs_store)} Records)")
    print("=" * 60)

if __name__ == "__main__":
    main()