"""
validate_and_pick_threshold.py — Score labeled validation data through the
trained Isolation Forest and pick a real operating threshold, instead of
relying on sklearn's `contamination` guess (which has no ground truth to be
checked against at training time, since training is benign-only).

What this does:
  1. Loads the model bundle from models/iso_forest.pkl (fitted on benign-only
     Datasets/Train_iso).
  2. Loads a VALIDATION set: benign holdout (unseen users) + one attack file
     (attacks_fast by default) -- NOT attacks_slow, which stays untouched as
     the final test set. Never tune the threshold against your final test data.
  3. Normalizes + featurizes validation data the same way training data was
     featurized, reusing the model's fitted label encoders (is_training=False)
     so categorical encoding is identical to what the model was trained on,
     and CORRECTLY realigned back to the original event order (see
     scoring_utils.py -- extract_features() reorders rows internally).
  4. Scores every row with decision_function(), sweeps candidate thresholds
     against real ground truth, and reports precision/recall/F1 across the
     sweep plus the overall PR-AUC.
  5. Picks the threshold that maximizes F1 (or hits a precision floor you
     set), and saves it into the model bundle for future scoring to use
     instead of sklearn's internal contamination-based offset_.

Ground truth: uses threat_category != "Normal" (not the raw anomaly_flag),
since generate_benign.py deliberately flags a small honest fraction of benign
off-hours activity as anomaly_flag=True / threat_category="Normal" -- that's
an unusual-but-legitimate oddity, not an actual attack the model should be
scored against.

Usage:
  python models/validate_and_pick_threshold.py \
      --model models/iso_forest.pkl \
      --holdout-dir Datasets/holdout \
      --attack-dir Datasets/attacks_fast
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from scoring_utils import load_normalize_and_label, featurize_and_align, sweep_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/iso_forest.pkl")
    ap.add_argument("--holdout-dir", default="Datasets/holdout")
    ap.add_argument("--attack-dir", default="Datasets/attacks_fast")
    ap.add_argument("--out", default=None,
                     help="where to save the updated bundle (default: overwrite --model)")
    ap.add_argument("--metric", choices=["f1", "precision_floor"], default="f1")
    ap.add_argument("--precision-floor", type=float, default=0.9,
                     help="if --metric precision_floor: pick the highest-recall "
                          "threshold with precision >= this value")
    ap.add_argument("--curve-out", default="models/pr_curve.csv",
                     help="CSV of the full precision/recall/threshold sweep, for "
                          "plotting a PR curve in the paper")
    args = ap.parse_args()

    print(f"Loading model bundle from {args.model} ...")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

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

    print("Loading + normalizing validation data ...")
    unified_h, y_h, _ = load_normalize_and_label({k: str(v) for k, v in holdout_paths.items()})
    unified_a, y_a, _ = load_normalize_and_label({k: str(v) for k, v in attack_paths.items()})

    unified = unified_h + unified_a
    y_true = np.concatenate([y_h, y_a])
    print(f"Validation set: {len(unified)} events "
          f"({int(y_true.sum())} genuine attack events, {len(unified) - int(y_true.sum())} benign)")

    X = featurize_and_align(unified, extractor, feature_columns)

    # Higher score == more anomalous. sklearn's decision_function() is LOWER
    # (more negative) for anomalies, so flip the sign for use with
    # precision_recall_curve, which expects higher score == more positive.
    anomaly_score = -model.decision_function(X)

    # Single shared sweep path (scoring_utils.sweep_threshold) -- same code the
    # risk-score threshold now uses, so the "sweep against labels, not a benign
    # quantile" discipline lives in exactly one place.
    metric = "precision_floor" if args.metric == "precision_floor" else "f1"
    sweep = sweep_threshold(anomaly_score, y_true, metric=metric,
                            precision_floor=args.precision_floor)
    pr_auc = sweep["pr_auc"]
    thresholds = sweep["thresholds"]
    p, r, f1 = sweep["precision_curve"], sweep["recall_curve"], sweep["f1_curve"]
    print(f"\nPR-AUC: {pr_auc:.4f}  (threshold-independent -- report this number in the paper)")

    if args.metric == "f1":
        print("\nBest-F1 operating point:")
    else:
        if not (p >= args.precision_floor).any():
            print(f"\n(no threshold reaches precision >= {args.precision_floor}; "
                  f"max precision available: {p.max():.4f} -- picking max-precision point)")
        print(f"\nHighest-recall point with precision >= {args.precision_floor}:")

    chosen_threshold = sweep["threshold"]
    print(f"  threshold (on -decision_function): {chosen_threshold:.4f}")
    print(f"  precision: {sweep['precision']:.4f}  recall: {sweep['recall']:.4f}  f1: {sweep['f1']:.4f}")

    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>8}")
    step = max(len(thresholds) // 15, 1)
    for i in range(0, len(thresholds), step):
        print(f"{thresholds[i]:>10.4f} {p[i]:>10.4f} {r[i]:>10.4f} {f1[i]:>8.4f}")

    curve_path = Path(args.curve_out)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curve_path, "w") as f:
        f.write("threshold,precision,recall,f1\n")
        for i in range(len(thresholds)):
            f.write(f"{thresholds[i]},{p[i]},{r[i]},{f1[i]}\n")
    print(f"\nFull PR sweep saved -> {curve_path} (for plotting a PR curve in the paper)")

    bundle["decision_threshold"] = chosen_threshold
    bundle["validation_pr_auc"] = pr_auc
    bundle["validation_precision"] = sweep["precision"]
    bundle["validation_recall"] = sweep["recall"]
    bundle["validation_f1"] = sweep["f1"]
    bundle["validation_source"] = {"holdout_dir": str(holdout_dir), "attack_dir": str(attack_dir)}

    out_path = Path(args.out) if args.out else Path(args.model)
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved updated bundle (model + chosen threshold) -> {out_path}")


if __name__ == "__main__":
    main()
