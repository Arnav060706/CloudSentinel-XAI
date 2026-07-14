"""
train_isolation_forest.py — Fit the unsupervised anomaly detector on benign-only
multi-cloud IAM logs.

IMPORTANT: --train-dir must point at BENIGN-ONLY data (the `train` split —
e.g. dataset/raw/train/, never dataset/merged/ which also contains attacks).
Isolation Forest learns "what normal looks like"; training on contaminated
data defeats the point.

Usage (run from the repo root):
  python models/train_isolation_forest.py --train-dir dataset/raw/train --out models
"""
import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

# Repo root on sys.path so `app.*` imports resolve when run as a plain script
# (python models/train_isolation_forest.py) instead of `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import IsolationForest

from app.parser_normalizer.src.pipeline import ParserPipeline
from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor


def load_and_normalize(paths: dict) -> list[dict]:
    pipeline = ParserPipeline()
    unified = []
    for source_cloud, path in paths.items():
        with open(path, "r") as f:
            raw_logs = json.load(f)
        for raw_log in raw_logs:
            normalized = pipeline.process_log(raw_log, source_cloud=source_cloud)
            if normalized is not None:
                unified.append(normalized.model_dump(exclude={"raw_log"}))
    if pipeline.failed_logs_store:
        print(f"WARNING: {len(pipeline.failed_logs_store)} logs failed normalization "
              f"and were skipped.")
    return unified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True,
                     help="dir containing aws_benign.json, azure_benign.json, "
                          "gcp_benign.json (BENIGN TRAIN split only)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent),
                     help="output dir for the model bundle")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--contamination", default="auto",
                     help="'auto' or a float like 0.03; only affects .predict()'s "
                          "binary threshold, not the continuous decision_function score")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    paths = {
        "AWS": train_dir / "aws_benign.json",
        "AZURE": train_dir / "azure_benign.json",
        "GCP": train_dir / "gcp_benign.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit(f"Missing benign log files: {missing}")

    print(f"Loading + normalizing benign logs from {train_dir} ...")
    unified = load_and_normalize({k: str(v) for k, v in paths.items()})
    print(f"Normalized {len(unified)} benign events.")

    extractor = MLFeatureExtractor()
    X, y = extractor.extract_features(unified, is_training=True)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Columns: {list(X.columns)}")

    contamination = args.contamination
    try:
        contamination = float(contamination)
    except ValueError:
        pass  # keep "auto"

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=contamination,
        random_state=args.random_state,
        n_jobs=-1,
    )
    print(f"\nFitting IsolationForest (n_estimators={args.n_estimators}, "
          f"contamination={contamination}) ...")
    model.fit(X)

    # Self-check on the training data itself. This is NOT a real evaluation
    # (that needs the held-out + attack data scored separately) -- it's only
    # a sanity check that fitting worked and scores have a sane spread.
    scores = model.decision_function(X)
    preds = model.predict(X)
    n_flagged = int((preds == -1).sum())
    print(f"\nTraining-set self-check (sanity only, not a real evaluation):")
    print(f"  {n_flagged}/{len(X)} rows flagged anomalous ({100 * n_flagged / len(X):.2f}%)")
    print(f"  decision_function range: [{scores.min():.4f}, {scores.max():.4f}], "
          f"mean={scores.mean():.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "label_encoders": extractor.label_encoders,
        "feature_columns": list(X.columns),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_rows": len(X),
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "random_state": args.random_state,
        "source_dir": str(train_dir),
    }
    out_path = out_dir / "iso_forest.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle -> {out_path}")


if __name__ == "__main__":
    main()
