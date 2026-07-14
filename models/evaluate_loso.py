"""
evaluate_loso.py — Phase 3: leave-one-scenario-out generalization test.

The XGBoost classifier scores 0.91 accuracy, but `attacks_fast` and
`attacks_slow` are both rendered from the SAME 4 `attack_scenarios.py` templates
(different seeds/victims/pacing only). So 0.91 measures "recognizes which of 4
KNOWN scripted scenarios a step belongs to", not generalization to an unseen
technique. LOSO measures the latter: train on 3 scenarios, test on the 4th,
never seen during training.

FRAMING (important): true LOSO cannot measure MULTI-CLASS accuracy on the
held-out scenario -- its `threat_category` labels may not exist in the training
folds at all (e.g. BruteForce / SuspiciousServiceAccountCreation live only in
the cross-cloud APT scenario). The honest, paper-defensible question is BINARY
generalization: does `1 - P(Normal)` rank the held-out scenario's attack events
above benign? That's also exactly the signal the risk engine consumes, so it's
the operationally relevant number. Multi-class "which bucket did unseen events
fall into" is reported too, but only descriptively.

Per fold: train XGBoost (same hyperparameters as train_xgboost.py, via
featurize_and_align) on the OTHER 3 scenarios' attack events + all benign-noise
events; test on the held-out scenario's attack events + a disjoint benign
sample. Reports per-fold and mean binary ROC-AUC / PR-AUC on `1 - P(Normal)`.

Usage:
  python models/evaluate_loso.py --attack-dir Datasets/attacks_fast \
      --benign-dir Datasets/holdout --out models/loso_results.csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from scoring_utils import load_normalize_and_label, featurize_and_align, discover_paths


def load_scenario_map(attack_dir):
    """event_id -> scenario, from the attack split's attack_labels.csv (which
    already carries both columns -- no dataset regeneration needed)."""
    m = {}
    with open(Path(attack_dir) / "attack_labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            m[r["event_id"]] = r["scenario"]
    return m


def _reindex(events):
    """Fresh copies with sequential __eval_row_id__ 0..n-1. REQUIRED: the events
    come from different load_normalize_and_label() calls, each of which numbered
    __eval_row_id__ from 0, so concatenating them produces DUPLICATE ids that
    would break featurize_and_align()'s argsort realignment. Reassigning makes
    them unique and contiguous again."""
    return [{**dict(e), "__eval_row_id__": j} for j, e in enumerate(events)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-dir", default="Datasets/attacks_fast",
                    help="attack split providing the per-scenario attack events + "
                         "benign-noise training negatives")
    ap.add_argument("--benign-dir", default="Datasets/holdout",
                    help="benign split used as the DISJOINT test-negative sample "
                         "(Normal class in the test folds)")
    ap.add_argument("--out", default="models/loso_results.csv")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="'cpu' (default) or 'cuda'")
    args = ap.parse_args()

    print(f"Loading attack split {args.attack_dir} + benign {args.benign_dir} ...")
    unified_a, _, meta_a = load_normalize_and_label(discover_paths(args.attack_dir, "attack"))
    unified_b, _, meta_b = load_normalize_and_label(discover_paths(args.benign_dir, "benign"))
    scen_map = load_scenario_map(args.attack_dir)

    scen = [scen_map.get(m["event_id"], "benign_noise") for m in meta_a]
    tcat = [m["threat_category"] for m in meta_a]
    scenarios = sorted({s for s, t in zip(scen, tcat)
                        if s != "benign_noise" and t not in ("Normal", "", None)})
    train_normal_idx = [i for i, s in enumerate(scen) if s == "benign_noise"]
    print(f"{len(scenarios)} scenarios: {scenarios}")
    print(f"benign-noise training negatives: {len(train_normal_idx)}; "
          f"benign test negatives: {len(unified_b)}")

    rows = []
    for held in scenarios:
        train_atk_idx = [i for i, (s, t) in enumerate(zip(scen, tcat))
                         if s not in (held, "benign_noise") and t not in ("Normal", "", None)]
        test_atk_idx = [i for i, (s, t) in enumerate(zip(scen, tcat))
                        if s == held and t not in ("Normal", "", None)]

        train_events = _reindex([unified_a[i] for i in train_atk_idx]
                                + [unified_a[i] for i in train_normal_idx])
        train_labels = np.array([tcat[i] for i in train_atk_idx]
                                + ["Normal"] * len(train_normal_idx))
        test_events = _reindex([unified_a[i] for i in test_atk_idx] + list(unified_b))
        test_y = np.array([1] * len(test_atk_idx) + [0] * len(unified_b))

        extractor = MLFeatureExtractor()
        X_train = featurize_and_align(train_events, extractor, feature_columns=None,
                                      is_training=True, include_labeled_only_features=True)
        fcols = list(X_train.columns)
        X_test = featurize_and_align(test_events, extractor, feature_columns=fcols,
                                     is_training=False, include_labeled_only_features=True)

        ce = LabelEncoder()
        y_train = ce.fit_transform(train_labels)
        sample_weight = compute_sample_weight("balanced", y_train)

        model = XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            objective="multi:softprob", random_state=args.random_state,
            eval_metric="mlogloss", device=args.device, tree_method="hist",
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

        proba = model.predict_proba(X_test)
        normal_idx = list(ce.classes_).index("Normal")
        attack_score = 1.0 - proba[:, normal_idx]

        roc = roc_auc_score(test_y, attack_score)
        pr = average_precision_score(test_y, attack_score)

        # Secondary/descriptive: where do the UNSEEN scenario's attack events get
        # bucketed? "misclassified as a different ATTACK" is a qualitatively
        # better failure than "classified Normal".
        held_pred = model.predict(X_test[:len(test_atk_idx)])
        pred_names = [ce.classes_[p] for p in held_pred]
        as_normal = sum(1 for p in pred_names if p == "Normal")
        as_attack = len(pred_names) - as_normal
        top_bucket = Counter(pred_names).most_common(3)

        base_rate = test_y.mean()
        rows.append({
            "held_out_scenario": held,
            "n_train_attack": len(train_atk_idx),
            "n_test_attack": len(test_atk_idx),
            "n_test_benign": len(unified_b),
            "test_base_rate": round(float(base_rate), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "unseen_flagged_not_normal": as_attack,
            "unseen_as_normal": as_normal,
            "unseen_flag_rate": round(as_attack / max(len(pred_names), 1), 4),
            "top_predicted_buckets": ";".join(f"{n}:{c}" for n, c in top_bucket),
        })
        print(f"\n[fold] held out = {held}")
        print(f"  train attacks={len(train_atk_idx)}  test attacks={len(test_atk_idx)}  "
              f"base_rate={base_rate:.3f}")
        print(f"  binary ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
        print(f"  unseen events flagged not-Normal: {as_attack}/{len(pred_names)} "
              f"({100*as_attack/max(len(pred_names),1):.0f}%); buckets={top_bucket}")

    mean_roc = float(np.mean([r["roc_auc"] for r in rows]))
    mean_pr = float(np.mean([r["pr_auc"] for r in rows]))

    print("\n" + "=" * 78)
    print(f"{'held-out scenario':<38}{'ROC':>8}{'PR-AUC':>9}{'flag%':>8}")
    for r in rows:
        print(f"{r['held_out_scenario']:<38}{r['roc_auc']:>8.4f}{r['pr_auc']:>9.4f}"
              f"{100*r['unseen_flag_rate']:>7.0f}%")
    print("-" * 78)
    print(f"{'MEAN (binary generalization)':<38}{mean_roc:>8.4f}{mean_pr:>9.4f}")
    print("=" * 78)
    print("Compare against the 0.91 IN-DISTRIBUTION multi-class accuracy: that number "
          "is scenario-templates-seen-during-training; the mean above is held-out-technique.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        w.writerow({"held_out_scenario": "MEAN", "roc_auc": round(mean_roc, 4),
                    "pr_auc": round(mean_pr, 4)})
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
