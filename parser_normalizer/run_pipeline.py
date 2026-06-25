import json
from src.pipeline import ParserPipeline

def load_json(filepath):
    """Safely load JSON logs."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {filepath} not found. Skipping...")
        return []

def main():
    pipeline = ParserPipeline()

    print("=" * 70)
    print("STARTING TRI-CLOUD LOG INGESTION & NORMALIZATION PIPELINE (SAMPLE DATA)")
    print("=" * 70)

    # 1. Load logs using the exact filenames from your screenshot
    aws_logs = load_json("mock_data/sample_logs_aws.json")
    azure_logs = load_json("mock_data/sample_azure.json")
    gcp_logs = load_json("mock_data/sample_gcp.json")

    if not any([aws_logs, azure_logs, gcp_logs]):
        print("ERROR: No logs found. Please check your mock_data folder.")
        return

    # 2. Combine into a unified stream
    unified_stream = []
    
    for log in aws_logs:
        payload = log.get("payload", log) if isinstance(log, dict) and "payload" in log else log
        unified_stream.append(("AWS", payload))
        
    for log in azure_logs:
        unified_stream.append(("AZURE", log))
        
    for log in gcp_logs:
        unified_stream.append(("GCP", log))

    print(f"Loaded {len(aws_logs)} AWS, {len(azure_logs)} Azure, and {len(gcp_logs)} GCP logs.\n")

    successful_logs = []

    # 3. Process Stream
    for idx, (cloud_source, raw_log) in enumerate(unified_stream, 1):
        print(f"\n[Processing Log #{idx}] Routing to: {cloud_source} Parser")
        
        normalized_record = pipeline.process_log(cloud_source, raw_log)
        
        if normalized_record:
            print("Status: SUCCESS")
            print(normalized_record.model_dump_json(indent=2, exclude={"raw_log"}))
            successful_logs.append(normalized_record)
        else:
            print("Status: FAILED (Added to failed logs store)")

    # 4. Final Summary
    print("\n" + "=" * 70)
    print("TRI-CLOUD PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Logs Processed: {len(unified_stream)}")
    print(f"Successful Validations: {len(successful_logs)}")
    print(f"Failed Validations: {len(pipeline.failed_logs_store)}")
    
    if pipeline.failed_logs_store:
        print("\n--- Failed Logs Details ---")
        for fail in pipeline.failed_logs_store:
            print(f"Error: {fail['error']}")

if __name__ == "__main__":
    main()