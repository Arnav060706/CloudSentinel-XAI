import json
from src.pipeline import ParserPipeline

def main():
    pipeline = ParserPipeline()

    print("=" * 70)
    print("STARTING TRI-CLOUD LOG INGESTION & NORMALIZATION PIPELINE (SAMPLE DATA)")
    print("=" * 70)

    # Loading mock logs from json file
    try:
        with open("mock_data/sample_logs.json", "r") as f:
            mock_entries = json.load(f)
    except FileNotFoundError:
        print("ERROR: mock_data/sample_logs.json not found. Please create it first.")
        return

    # Iterating and processing each log through the pipeline
    for idx, entry in enumerate(mock_entries, 1):
        cloud_source = entry.get("source", "UNKNOWN")
        payload = entry.get("payload", {})
        
        print(f"\n[Processing Log #{idx}] Source: {cloud_source}")
        
        # Run through parser, normalizer, and Pydantic validator
        normalized_record = pipeline.process_log(cloud_source, payload)
        
        if normalized_record:
            print("Status: SUCCESS ✅")
            # model_dump_json() serializes the Pydantic model into a clean JSON string
            print(normalized_record.model_dump_json(indent=2))
        else:
            print("Status: FAILED ❌ (To be sent to failed logs store (Yet to be configured))")

    # 3. Inspect the validation failure store
    print("\n" + "=" * 60)
    print(f"PIPELINE SUMMARY - FAILED LOGS STORE ({len(pipeline.failed_logs_store)} Records)")
    print("=" * 60)
    print(json.dumps(pipeline.failed_logs_store, indent=2))

if __name__ == "__main__":
    main()