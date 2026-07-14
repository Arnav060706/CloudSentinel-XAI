import json

from app.parser_normalizer.src.pipeline import ParserPipeline
from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor

DATASET_PATHS = {
    "AWS":   r"C:\Users\shrey\OneDrive\Documents\CCNCS\CloudSentinel-XAI\Datasets\Train_iso\aws_benign.json",
    "AZURE": r"C:\Users\shrey\OneDrive\Documents\CCNCS\CloudSentinel-XAI\Datasets\Train_iso\azure_benign.json",
    "GCP":   r"C:\Users\shrey\OneDrive\Documents\CCNCS\CloudSentinel-XAI\Datasets\Train_iso\gcp_benign.json",
}


def load_and_normalize(paths: dict[str, str]) -> list[dict]:
    """Load raw per-cloud log files and normalize them into the unified schema."""
    pipeline = ParserPipeline()
    unified = []

    for source_cloud, path in paths.items():
        with open(path, "r") as f:
            raw_logs = json.load(f)

        for raw_log in raw_logs:
            normalized = pipeline.process_log(raw_log, source_cloud=source_cloud)
            if normalized is not None:
                unified.append(normalized.model_dump(exclude={"raw_log"}))

    print(f"Normalized {len(unified)} logs. Failed: {len(pipeline.failed_logs_store)}")
    return unified


if __name__ == "__main__":
    unified = load_and_normalize(DATASET_PATHS)
    X, y = MLFeatureExtractor().extract_features(unified, is_training=True)

    print(f"\nFeature matrix X: {X.shape}")
    print(f"Target matrix y: {y.shape}")
    print("\nFeature columns:", list(X.columns))
