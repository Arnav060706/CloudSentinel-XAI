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
    print("STARTING AUTO-DETECT UNIFIED LOG INGESTION PIPELINE")
    print("=" * 70)

    # 1. Load the single unified file
    unified_logs = load_json("mock_data/unified_datastream.json")

    if not unified_logs:
        print("ERROR: No logs found in unified_datastream.json.")
        return

    print(f"Loaded {len(unified_logs)} raw, native logs.\n")

    successful_logs = []

    # 2. Process Stream (No manual tagging needed!)
    for idx, raw_log in enumerate(unified_logs, 1):
        print(f"\n[Processing Log #{idx}]")
        
        # Notice we only pass the raw_log now. The pipeline figures out the rest.
        normalized_record = pipeline.process_log(raw_log)
        
        if normalized_record:
            print(f"Detected Source: {normalized_record.source_cloud}")
            print("Status: SUCCESS")
            print(normalized_record.model_dump_json(indent=2, exclude={"raw_log"}))
            successful_logs.append(normalized_record)
        else:
            print("Status: FAILED (Added to failed logs store)")

    # 3. Final Summary
    print("\n" + "=" * 70)
    print("AUTO-DETECT PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Logs Processed: {len(unified_logs)}")
    print(f"Successful Validations: {len(successful_logs)}")
    print(f"Failed Validations: {len(pipeline.failed_logs_store)}")

if __name__ == "__main__":
    main()