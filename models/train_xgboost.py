"""
train_xgboost.py — Train the supervised multi-class attack-type classifier.

Target: threat_category (BruteForce, PrivilegeEscalation, Reconnaissance,
CredentialCreation, DefenseEvasion, SuspiciousLogin, AccountCompromised,
UnauthorizedAccessAttempt, SuspiciousServiceAccountCreation, and "Normal" for
benign) -- classifies WHICH TYPE of activity an event represents. Runs in
parallel with the Isolation Forest per ml_inference.py's existing design
(every event gets both an anomaly_score AND a predicted_phase).

Train: Datasets/attacks_fast (labeled attacks + benign noise, mixed). Pure
benign-only data (Train_iso) is deliberately NOT used here -- a multi-class
classifier needs multiple classes to discriminate between; a benign-only
training set is exactly what the (unsupervised) Isolation Forest is for.
Test: Datasets/attacks_slow -- reserved and untouched until now.

Uses include_labeled_only_features=True (see feature_extractor.py) since
geo_country/is_known_proxy_or_tor/device_compliant_status/is_internal_ip have
real, confirmed variance on labeled attack data (e.g. is_known_proxy_or_tor
is 41 True / 328 False on attacks_fast) unlike the benign-only Isolation
Forest training set, where they're zero-variance by construction.

Usage:
  python models/train_xgboost.py --train-dir Datasets/attacks_fast \
      --test-dir Datasets/attacks_slow --out models
"""
import argparse
import datetime as dt
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from scoring_utils import featurize_and_align, load_normalize_and_label


def load_attack_dir(attack_dir: Path):
    paths = {
        "AWS": attack_dir / "aws_attack.json",
        "AZURE": attack_dir / "azure_attack.json",
        "GCP": attack_dir / "gcp_attack.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit(f"Missing files: {missing}")
    unified, _, meta = load_normalize_and_label({k: str(v) for k, v in paths.items()})
    y = np.array([m["threat_category"] for m in meta])
    return unified, y


def compute_background_sample(X_train, n: int, seed: int) -> np.ndarray:
    """Phase 6: a reference sample of ENGINEERED training rows for the SHAP
    faithfulness deletion test. "Deleting" a feature means replacing it with a
    background/reference value (its typical value), NOT zeroing it -- 0 is a
    meaningful value for encoded categoricals and counts. xai_engine derives a
    per-column reference (median) from this. Aligned to feature_columns order."""
    n = min(int(n), len(X_train))
    return X_train.sample(n=n, random_state=seed).to_numpy(dtype=float)


def add_background_to_existing_bundle(train_dir: Path, out_dir: Path, n: int):
    """Patch an existing xgboost_classifier.pkl with a background_sample WITHOUT
    retraining -- the model stays byte-identical (so every recorded ablation /
    accuracy number remains valid), we only add training-data statistics. The
    engineered matrix is reproduced from the SAME training data using the
    bundle's already-fitted encoders (is_training=False)."""
    bundle_path = out_dir / "xgboost_classifier.pkl"
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    unified, _ = load_attack_dir(train_dir)
    extractor = MLFeatureExtractor()
    extractor.label_encoders = bundle["label_encoders"]
    X_train = featurize_and_align(
        unified, extractor, feature_columns=bundle["feature_columns"],
        is_training=False, include_labeled_only_features=True,
    )
    bundle["background_sample"] = compute_background_sample(X_train, n, bundle["random_state"])
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Patched {bundle_path} with background_sample "
          f"{bundle['background_sample'].shape} (model unchanged).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="Datasets/attacks_fast")
    ap.add_argument("--test-dir", default="Datasets/attacks_slow")
    ap.add_argument("--out", default="models")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--device", default="cuda", help="'cuda' (default, RTX 3050) or 'cpu'")
    ap.add_argument("--background-n", type=int, default=100,
                    help="rows sampled into the SHAP-deletion background_sample")
    ap.add_argument("--background-only", action="store_true",
                    help="don't retrain; just add background_sample to the existing "
                         "bundle (model unchanged, all recorded numbers stay valid)")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if args.background_only:
        add_background_to_existing_bundle(train_dir, Path(args.out), args.background_n)
        return

    print(f"Loading + normalizing training data from {train_dir} ...")
    unified_train, y_train_raw = load_attack_dir(train_dir)
    classes_train, counts_train = np.unique(y_train_raw, return_counts=True)
    print(f"{len(unified_train)} events. Class distribution: "
          f"{dict(zip(classes_train, counts_train))}")

    print(f"\nLoading + normalizing test data from {test_dir} (reserved, untouched until now) ...")
    unified_test, y_test_raw = load_attack_dir(test_dir)
    classes_test, counts_test = np.unique(y_test_raw, return_counts=True)
    print(f"{len(unified_test)} events. Class distribution: "
          f"{dict(zip(classes_test, counts_test))}")

    extractor = MLFeatureExtractor()
    # ALWAYS go through featurize_and_align, for training too -- see its
    # docstring. extract_features() sorts rows by timestamp internally, so
    # calling it directly and zipping the output against y_train_raw (built
    # in original file-load order) would silently misalign labels and
    # features, exactly the bug this module exists to prevent.
    X_train = featurize_and_align(
        unified_train, extractor, feature_columns=None, is_training=True,
        include_labeled_only_features=True,
    )
    feature_columns = list(X_train.columns)
    print(f"\nFeature matrix: {X_train.shape[0]} rows x {X_train.shape[1]} features")
    print(f"Columns: {feature_columns}")

    # Test set encoded with the SAME fitted label encoders as training
    # (is_training=False), reindexed to the exact same column set/order.
    X_test = featurize_and_align(
        unified_test, extractor, feature_columns=feature_columns, is_training=False,
        include_labeled_only_features=True,
    )

    class_encoder = LabelEncoder()
    y_train = class_encoder.fit_transform(y_train_raw)

    known = set(class_encoder.classes_)
    test_mask = np.array([c in known for c in y_test_raw])
    if not test_mask.all():
        dropped = sorted(set(y_test_raw[~test_mask]))
        print(f"\nWARNING: dropping {(~test_mask).sum()} test rows with classes "
              f"unseen in training: {dropped}")
    X_test_eval = X_test[test_mask].reset_index(drop=True)
    y_test = class_encoder.transform(y_test_raw[test_mask])

    # Class counts are quite imbalanced (e.g. SuspiciousServiceAccountCreation
    # vs PrivilegeEscalation) -- balanced sample weights so XGBoost doesn't
    # just learn to ignore the rare classes.
    sample_weight = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        objective="multi:softprob",
        random_state=args.random_state,
        eval_metric="mlogloss",
        device=args.device,
        tree_method="hist",
    )
    print(f"\nFitting XGBClassifier ({len(class_encoder.classes_)} classes, "
          f"{X_train.shape[1]} features, balanced sample weights, device={args.device}) ...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    print("\nEvaluating on held-out test set (Datasets/attacks_slow) ...")
    y_pred = model.predict(X_test_eval)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}   Macro F1: {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(
        y_test, y_pred, labels=range(len(class_encoder.classes_)),
        target_names=class_encoder.classes_, zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_encoder.classes_)))
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{c[:12]:>14s}" for c in class_encoder.classes_)
    print(f"{'':25s}{header}")
    for i, row in enumerate(cm):
        print(f"{class_encoder.classes_[i]:25s}" + "".join(f"{v:>14d}" for v in row))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "label_encoders": extractor.label_encoders,
        "class_encoder": class_encoder,
        "class_names": list(class_encoder.classes_),
        "feature_columns": feature_columns,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_rows": len(X_train),
        "test_rows": len(X_test_eval),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "source_train_dir": str(train_dir),
        "source_test_dir": str(test_dir),
        # Phase 6: reference sample for the SHAP faithfulness deletion test.
        "background_sample": compute_background_sample(X_train, args.background_n, args.random_state),
    }
    out_path = out_dir / "xgboost_classifier.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle -> {out_path}")


if __name__ == "__main__":
    main()
