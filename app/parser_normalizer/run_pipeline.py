import json
from src.pipeline import ParserPipeline
from src.feature_extractor import MLFeatureExtractor

def load_json(filepath):
    """Safely load JSON logs."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {filepath} not found. Skipping...")
        return []

def main():
    # Initialize both the Data Engineering and Machine Learning engines
    pipeline = ParserPipeline()
    extractor = MLFeatureExtractor()

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
    ml_ready_logs = [] # We need dictionaries for the ML Extractor

    # 2. Process Stream
    for idx, raw_log in enumerate(unified_logs, 1):
        print(f"\n[Processing Log #{idx}]")
        
        normalized_record = pipeline.process_log(raw_log)
        
        if normalized_record:
            print(f"Detected Source: {normalized_record.source_cloud}")
            print("Status: SUCCESS")
            
            successful_logs.append(normalized_record)
            # Convert the Pydantic model back to a standard dictionary for ML
            # Exclude raw_log to keep the ML input clean
            ml_ready_logs.append(normalized_record.model_dump(exclude={"raw_log"})) 
        else:
            print("Status: FAILED (Added to failed logs store)")

    # 3. Export Normalized Logs
    if ml_ready_logs:
        with open("normalized_logs.json", "w") as f:
            # Using default=str to handle datetime serialization
            json.dump(ml_ready_logs, f, indent=4, default=str)
        print(f"\n[INFO] Successfully exported {len(ml_ready_logs)} normalized logs to 'normalized_logs.json'")

    # 4. Final Summary for Data Engineering
    print("\n" + "=" * 70)
    print("AUTO-DETECT PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Total Logs Processed: {len(unified_logs)}")
    print(f"Successful Validations: {len(successful_logs)}")
    print(f"Failed Validations: {len(pipeline.failed_logs_store)}")

    # 5. Machine Learning Feature Extraction Phase
    if ml_ready_logs:
        print("\n" + "=" * 70)
        print("MACHINE LEARNING FEATURE EXTRACTION")
        print("=" * 70)
        
        # Pass the clean dictionaries into our extractor
        # export_csv=True triggers the creation of features_X.csv and targets_y.csv
        X, y = extractor.extract_features(ml_ready_logs, is_training=True, export_csv=True)
        
        print(f"Successfully engineered {len(X.columns)} features for {len(X)} logs.")
        print("[INFO] Features exported to 'features_X.csv' and targets to 'targets_y.csv'")
        
        print("\n[+] The AI Feature Matrix (X) - Ready for Isolation Forest / XGBoost:")
        print(X.to_string())
        
        print("\n[+] The Target Variables (y) - Safely isolated from training data:")
        print(y.to_string())

if __name__ == "__main__":
    main()