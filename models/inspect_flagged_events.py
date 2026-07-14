"""
inspect_flagged_events.py — See exactly which logs the model flagged, and
compute a fuller metrics report than the training/validation scripts do.

What this does:
  1. Loads the model bundle (must already have `decision_threshold` set by
     validate_and_pick_threshold.py -- run that first).
  2. Scores a labeled dataset (default: the same holdout + attacks_fast
     validation set) through the model, with correct row alignment
     (scoring_utils.py -- do not zip labels against X by raw position).
  3. Writes EVERY scored event to a CSV (timestamp, cloud, user, action,
     anomaly_score, flagged, true label, threat category), sorted by
     anomaly_score descending -- open this file to literally see what was
     flagged and what wasn't, ranked by how suspicious the model found it.
  4. Prints an expanded metrics report at the model's chosen threshold:
     confusion matrix, accuracy, precision, recall, F1, false-positive rate,
     ROC-AUC, plus a per-threat-category and per-cloud breakdown so you can
     see which kinds of attacks are (and aren't) being caught.

Usage:
  # validation set (same data the threshold was picked on)
  python models/inspect_flagged_events.py

  # final untouched test set -- only run this once you're done tuning
  python models/inspect_flagged_events.py --attack-dir Datasets/attacks_slow \
      --out-csv models/scored_events_test.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from scoring_utils import load_normalize_and_label, featurize_and_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/iso_forest.pkl")
    ap.add_argument("--holdout-dir", default="Datasets/holdout")
    ap.add_argument("--attack-dir", default="Datasets/attacks_fast")
    ap.add_argument("--out-csv", default="models/scored_events.csv")
    ap.add_argument("--top-n", type=int, default=25,
                     help="how many highest-scored rows to print to the console")
    args = ap.parse_args()

    print(f"Loading model bundle from {args.model} ...")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = bundle.get("decision_threshold")
    if threshold is None:
        sys.exit("No decision_threshold in the model bundle -- run "
                  "validate_and_pick_threshold.py first.")

    extractor = MLFeatureExtractor()
    extractor.label_encoders = bundle["label_encoders"]

    holdout_dir = Path(args.holdout_dir)
    attack_dir = Path(args.attack_dir)
    holdout_paths = {
        "AWS": holdout_dir / "aws_benign_holdout.json",
        "AZURE": holdout_dir / "azure_benign_holdout.json",
        "GCP": holdout_dir / "gcp_benign_holdout.json",
    }
    attack_paths = {
        "AWS": attack_dir / "aws_attack.json",
        "AZURE": attack_dir / "azure_attack.json",
        "GCP": attack_dir / "gcp_attack.json",
    }
    for label, paths in (("holdout", holdout_paths), ("attack", attack_paths)):
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            sys.exit(f"Missing {label} files: {missing}")

    print("Loading + normalizing data ...")
    unified_h, y_h, meta_h = load_normalize_and_label({k: str(v) for k, v in holdout_paths.items()})
    unified_a, y_a, meta_a = load_normalize_and_label({k: str(v) for k, v in attack_paths.items()})

    unified = unified_h + unified_a
    y_true = np.concatenate([y_h, y_a])
    meta = meta_h + meta_a
    print(f"Scoring {len(unified)} events "
          f"({int(y_true.sum())} genuine attack events, {len(unified) - int(y_true.sum())} benign)")

    X = featurize_and_align(unified, extractor, feature_columns)
    anomaly_score = -model.decision_function(X)  # higher == more anomalous
    flagged = anomaly_score >= threshold

    report = pd.DataFrame(meta)
    report["anomaly_score"] = anomaly_score
    report["flagged"] = flagged
    report["true_label"] = np.where(y_true == 1, "ATTACK", "benign")
    report = report.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(f"\nFull scored event list -> {out_path} "
          f"(sorted by anomaly_score descending -- flagged events are at the top)")

    print(f"\nTop {args.top_n} most anomalous events:")
    cols = ["anomaly_score", "flagged", "true_label", "source_cloud", "user_id", "action", "threat_category"]
    with pd.option_context("display.max_colwidth", 40, "display.width", 200):
        print(report[cols].head(args.top_n).to_string(index=False))

    # ---------------------------------------------------------------- #
    # Metrics report at the model's chosen threshold
    # ---------------------------------------------------------------- #
    y_pred = flagged.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc = roc_auc_score(y_true, anomaly_score) if len(set(y_true)) > 1 else float("nan")

    print(f"\n{'=' * 60}\nMETRICS AT THRESHOLD {threshold:.4f}\n{'=' * 60}")
    print(f"Confusion matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall (TPR):        {recall:.4f}")
    print(f"F1:                  {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}  (threshold-independent)")

    print(f"\nDetection rate by threat_category (of ACTUAL attack events, "
          f"how many were flagged):")
    atk = report[report["true_label"] == "ATTACK"]
    by_cat = atk.groupby("threat_category")["flagged"].agg(["sum", "count"])
    by_cat["detection_rate"] = by_cat["sum"] / by_cat["count"]
    print(by_cat.rename(columns={"sum": "flagged", "count": "total"}).to_string())

    print(f"\nFalse positive rate by cloud (of BENIGN events, how many were "
          f"wrongly flagged):")
    ben = report[report["true_label"] == "benign"]
    by_cloud = ben.groupby("source_cloud")["flagged"].agg(["sum", "count"])
    by_cloud["fpr"] = by_cloud["sum"] / by_cloud["count"]
    print(by_cloud.rename(columns={"sum": "flagged", "count": "total"}).to_string())

    print(f"\n{'=' * 60}")
    print("Copy these numbers into models/README.md's results section.")


if __name__ == "__main__":
    main()
