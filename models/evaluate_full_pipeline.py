"""
evaluate_full_pipeline.py — Full ablation: which per-event signal (Isolation
Forest vs XGBoost) and which risk layer (isolated vs graph-stitched) actually
detect attacks best, on the SAME final evaluation set throughout.

IMPORTANT dataset choice: uses Datasets/holdout + Datasets/attacks_slow, NOT
attacks_fast. attacks_fast is XGBoost's TRAINING data -- reusing it here for
the XGBoost rows would be data leakage. Isolation Forest was never trained on
any attack data at all (benign-only), so it's fine to evaluate on
attacks_slow too, which lets every row in this table share one leak-free,
consistent evaluation set instead of comparing rows computed on different data.

Six rows, same ground truth throughout (B/C/D for each of IF and XGBoost):
  B1. IF + risk, no graph        -- risk formula alone on the IF signal, no
                                     identity stitching (isolated per-event
                                     "entity"). See models/README.md for the
                                     plain IF-only per-event baseline.
  C1. IF + risk + graph          -- IF signal through the full cross-cloud
                                     identity-stitching + risk pipeline.
  D1. IF + risk + ORACLE graph   -- Phase 1c: entity resolution keyed by the
                                     ground-truth actor (perfect stitching, zero
                                     merge errors) via the Tier-1 federation
                                     join; everything else identical to C1. D vs
                                     B isolates 'is the graph concept sound' from
                                     'are stitching errors the problem'.
  B2. XGBoost + risk, no graph   -- same risk formula, fed 1 - P(Normal) from
                                     XGBoost instead of the IF anomaly score.
  C2. XGBoost + risk + graph     -- XGBoost signal through the full pipeline.
  D2. XGBoost + risk + ORACLE graph -- oracle stitching on the XGBoost signal.
      risk_engine.py only reads "anomaly_score" (falling back to
      "phase_confidence" ONLY when anomaly_score is absent -- see
      risk_engine.py:221-223), so simply attaching both fields would let
      XGBoost's much stronger per-event signal (0.91 accuracy vs IF's 0.52
      ROC-AUC) get silently ignored. Instead, XGBoost's signal REPLACES
      anomaly_score for the B2/C2 rows: 1 - P(Normal) is XGBoost's own
      continuous "how likely is this ANY kind of attack" estimate, the same
      role anomaly_score already plays.

Two things had to be handled correctly for ANY of this to be meaningful --
see the B1/C1 rows' original development in git history / prior README
sections for how each was diagnosed:

1. MultiCloudGraphEngine's sliding window uses time.time() (real wall clock)
   for arrival bookkeeping, not the event's own timestamp -- correct for
   live streaming, wrong for REPLAYING historical data. This script patches
   time.time() (inside the graph_engine module specifically) to follow each
   event's own historical timestamp while iterating strictly in
   chronological order.

2. [Phase 2] The alert threshold is FROZEN by sweeping it against the labeled
   CALIBRATION set (holdout_cal + attacks_cal) via scoring_utils.sweep_threshold,
   then applied unchanged to the eval set -- never calibrated on the eval set.
   This replaces the earlier p99-of-benign quantile, which landed above every
   attack score despite real ranking separation (0 TP at ROC-AUC 0.68). Pass
   --cal-* to change the calibration split; if it equals the eval set the sweep
   is in-sample (used only for the on-cal verify, not the frozen number).

3. [Phase 1] The cross-cloud multiplier is baseline-aware: pass --baseline-dir /
   --cal-baseline-dir (a DISJOINT same-population benign split) so each identity's
   normal cloud set is learned first and only genuinely novel crossings amplify.
   Without it the multiplier inflates benign multi-cloud users -- see the README.

Usage:
  python models/evaluate_full_pipeline.py \
      --iso-model models/iso_forest.pkl \
      --xgb-model models/xgboost_classifier.pkl \
      --holdout-dir Datasets/holdout \
      --attack-dir Datasets/attacks_slow \
      --out models/ablation_results.csv
"""
import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.services.graph_engine as graph_engine_module
from app.services.graph_engine import MultiCloudGraphEngine
from app.services.risk_engine import RiskEngine
from app.parser_normalizer.src.feature_extractor import MLFeatureExtractor
from scoring_utils import (load_normalize_and_label, featurize_and_align,
                           discover_paths, sweep_threshold)


def _epoch(ts) -> float:
    """unified event timestamps are python datetime objects (from
    model_dump() on a pydantic datetime field). Handle that plus plain
    numbers/strings defensively."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def run_isolated(events_sorted):
    """Risk scoring with NO identity stitching -- each event is its own
    single-event entity (own cloud only, no cross-event window)."""
    risk = RiskEngine()
    return [
        risk.calculate_intensity(
            active_events=[e],
            lifetime_clouds=[e.get("source_cloud")],
            entity_id=e.get("principal"),
            principal_type=e.get("principal_type"),
        )
        for e in events_sorted
    ]


def run_full_pipeline(events_sorted, oracle_keys=None,
                      baseline_events_sorted=None, baseline_oracle_keys=None):
    """Full pipeline -- MultiCloudGraphEngine identity stitching feeds
    RiskEngine's cross-cloud multiplier. Patches time.time() so the 60s
    sliding window follows historical event time, not wall clock.

    oracle_keys (Phase 1c, row D): if given, an aligned list of GROUND-TRUTH
    principal keys, one per event. Each is injected as the event's
    `federation_id`, so the graph engine's Tier-1 federation join keys entities
    by ground truth -- PERFECT cross-cloud stitching, zero merge errors, while
    every other part of the pipeline (window, lifetime footprint, risk formula,
    calibration) stays byte-identical to row C. Comparing D vs B isolates
    'is the graph CONCEPT sound' from 'are stitching ERRORS the problem'.

    baseline_events_sorted (Phase 1 fix): if given, a DISJOINT benign set (same
    population, different events -- the holdout <-> holdout_cal pair) is replayed
    FIRST through the same graph, and each resulting entity's benign lifetime
    cloud footprint is recorded as its RiskEngine baseline. The cross-cloud
    multiplier then only fires for clouds an identity crosses BEYOND its normal
    set, instead of penalizing every benign multi-cloud user -- the root-cause
    fix from 'Phase 1: root-causing the graph regression' in models/README.md.
    Windows are reset between the warmup and scoring passes so no stale warmup
    event leaks into scoring."""
    graph = MultiCloudGraphEngine()
    risk = RiskEngine()

    class _Clock:
        now = 0.0

    clock = _Clock()
    real_time_time = graph_engine_module.time.time
    graph_engine_module.time.time = lambda: clock.now
    try:
        # --- Warmup pass: learn per-identity benign cloud baselines ----------
        if baseline_events_sorted is not None:
            for j, be in enumerate(baseline_events_sorted):
                clock.now = _epoch(be["timestamp"])
                if baseline_oracle_keys is not None:
                    be = dict(be)
                    be["federation_id"] = baseline_oracle_keys[j] or be.get("principal")
                graph.process_event(be)
            for eid, clouds in graph.entity_lifetime_footprints().items():
                risk.record_baseline(eid, clouds)
            graph.reset_windows()  # don't let warmup events count in scoring

        # --- Scoring pass ----------------------------------------------------
        results = []
        for i, e in enumerate(events_sorted):
            clock.now = _epoch(e["timestamp"])
            if oracle_keys is not None:
                e = dict(e)
                e["federation_id"] = oracle_keys[i] or e.get("principal")
            entity_id, active_events, is_new, method, lifetime_clouds = graph.process_event(e)
            r = risk.calculate_intensity(
                active_events=active_events,
                lifetime_clouds=lifetime_clouds,
                entity_id=entity_id,
                principal_type=e.get("principal_type"),
            )
            results.append(r)
        return results
    finally:
        graph_engine_module.time.time = real_time_time  # restore, don't leak the patch


def metrics_report(name, eval_y, eval_results, cal_y, cal_results, metric="f1"):
    """Phase 2: FREEZE the alert threshold by sweeping it on the labeled
    CALIBRATION set (scoring_utils.sweep_threshold), never the eval set, then
    evaluate on the eval set at that frozen threshold. This replaces the old
    p99-of-benign quantile, which landed above every attack score despite real
    ranking separation (0 TP at ROC-AUC 0.68 -- see models/README.md). Reports
    the frozen threshold + eval confusion matrix (threshold-dependent) and eval
    ROC/PR-AUC (threshold-independent). When cal is the eval set itself (same
    dirs), this is an in-sample threshold; the honest number comes from cal =
    the disjoint cal split, eval = the frozen attacks_slow set."""
    eval_scaled = np.array([r["scaled_score"] for r in eval_results])
    cal_scaled = np.array([r["scaled_score"] for r in cal_results])
    eval_y = np.asarray(eval_y)
    cal_y = np.asarray(cal_y)

    sweep = sweep_threshold(cal_scaled, cal_y, metric=metric)
    thr = sweep["threshold"]

    y_pred = (eval_scaled >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(eval_y, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / len(eval_y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc = roc_auc_score(eval_y, eval_scaled) if len(set(eval_y)) > 1 else float("nan")
    pr_auc = average_precision_score(eval_y, eval_scaled)

    return {
        "name": name, "calibrated_threshold": thr, "cal_f1": sweep["f1"],
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "roc_auc": roc_auc, "pr_auc": pr_auc,
    }


def prepare_set(holdout_dir, attack_dir, baseline_dir, iso_bundle, xgb_bundle):
    """Load + normalize + featurize a (benign, attack) split pair and return
    everything the row runners need: chronologically-sorted IF/XGBoost event
    lists (with anomaly_score attached), ground-truth keys, labels, and an
    optional disjoint benign warmup set for baseline learning (Phase 1)."""
    unified_h, y_h, meta_h = load_normalize_and_label(discover_paths(str(holdout_dir), "benign"))
    unified_a, y_a, meta_a = load_normalize_and_label(discover_paths(str(attack_dir), "attack"))
    unified = unified_h + unified_a
    y = np.concatenate([y_h, y_a])
    gt_keys = [m.get("gt_principal") for m in (meta_h + meta_a)]
    event_ids = [m.get("event_id") for m in (meta_h + meta_a)]

    iso_ex = MLFeatureExtractor(); iso_ex.label_encoders = iso_bundle["label_encoders"]
    X_iso = featurize_and_align(unified, iso_ex, iso_bundle["feature_columns"],
                                is_training=False, include_labeled_only_features=False)
    # 0.5 - decision_function keeps anomaly_score in the [0,1]-ish convention
    # risk_engine.py clamps to (see this file's earlier comment history).
    iso_anomaly = 0.5 - iso_bundle["model"].decision_function(X_iso)

    xgb_ex = MLFeatureExtractor(); xgb_ex.label_encoders = xgb_bundle["label_encoders"]
    X_xgb = featurize_and_align(unified, xgb_ex, xgb_bundle["feature_columns"],
                                is_training=False, include_labeled_only_features=True)
    proba = xgb_bundle["model"].predict_proba(X_xgb)
    normal_idx = list(xgb_bundle["class_encoder"].classes_).index("Normal")
    xgb_prob = 1.0 - proba[:, normal_idx]  # P(any attack class), not just argmax

    def build(scores):
        out = []
        for u, s in zip(unified, scores):
            e = dict(u); e["anomaly_score"] = float(s); e["principal"] = u.get("user_id", "")
            out.append(e)
        return out

    events_iso, events_xgb = build(iso_anomaly), build(xgb_prob)

    base_events = base_gt = None
    if baseline_dir:
        ub, _yb, mb = load_normalize_and_label(discover_paths(str(baseline_dir), "benign"))
        bl = []
        for u, m in zip(ub, mb):
            e = dict(u); e["principal"] = u.get("user_id", ""); e["anomaly_score"] = 0.0
            bl.append((e, m.get("gt_principal")))
        bl.sort(key=lambda t: _epoch(t[0]["timestamp"]))
        base_events = [t[0] for t in bl]
        base_gt = [t[1] for t in bl]

    order = sorted(range(len(unified)), key=lambda i: events_iso[i]["timestamp"])
    return {
        "iso": [events_iso[i] for i in order],
        "xgb": [events_xgb[i] for i in order],
        "gt": [gt_keys[i] for i in order],
        # Phase 7: per-event join key back to attack_labels.csv, for
        # campaign-level grouping (models/evaluate_campaign_recall.py).
        "event_id": [event_ids[i] for i in order],
        "y": y[order],
        "base_events": base_events, "base_gt": base_gt,
        "n": len(unified), "n_attack": int(y.sum()),
    }


def run_row(setd, signal, mode):
    """Score one (signal in {iso,xgb}) x (mode in {isolated,graph,oracle}) row on
    a prepared set. graph/oracle use the set's disjoint benign baseline warmup."""
    events = setd[signal]
    if mode == "isolated":
        return run_isolated(events)
    base, base_gt = setd["base_events"], setd["base_gt"]
    if mode == "graph":
        return run_full_pipeline(events, baseline_events_sorted=base)
    return run_full_pipeline(events, oracle_keys=setd["gt"],
                             baseline_events_sorted=base, baseline_oracle_keys=base_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-model", default="models/iso_forest.pkl")
    ap.add_argument("--xgb-model", default="models/xgboost_classifier.pkl")
    # EVAL set (reported ROC/PR-AUC + frozen-threshold confusion matrix).
    ap.add_argument("--holdout-dir", default="Datasets/holdout")
    ap.add_argument("--attack-dir", default="Datasets/attacks_slow",
                     help="MUST NOT be attacks_fast -- that's XGBoost's training data")
    ap.add_argument("--baseline-dir", default=None,
                     help="Benign dir to learn per-identity cloud baselines from "
                          "(Phase 1 fix), DISJOINT from --holdout-dir. For the frozen "
                          "eval use Datasets/holdout_cal.")
    # CALIBRATION set (threshold swept here, then frozen -- Phase 2). Never the eval set.
    ap.add_argument("--cal-holdout-dir", default="Datasets/holdout_cal")
    ap.add_argument("--cal-attack-dir", default="Datasets/attacks_cal",
                     help="labeled attack split used ONLY to sweep the threshold")
    ap.add_argument("--cal-baseline-dir", default=None,
                     help="baseline source for the cal set, DISJOINT from "
                          "--cal-holdout-dir. For the frozen eval use Datasets/holdout.")
    ap.add_argument("--metric", choices=["f1", "precision_floor"], default="f1")
    ap.add_argument("--out", default="models/ablation_results.csv")
    args = ap.parse_args()

    for adir in (args.attack_dir, args.cal_attack_dir):
        if "attacks_fast" in str(adir):
            sys.exit("Refusing to use attacks_fast (XGBoost training data).")
    for bdir, hdir, what in ((args.baseline_dir, args.holdout_dir, "eval"),
                             (args.cal_baseline_dir, args.cal_holdout_dir, "cal")):
        if bdir and Path(bdir).resolve() == Path(hdir).resolve():
            sys.exit(f"--{'cal-' if what=='cal' else ''}baseline-dir must be DISJOINT "
                     f"from the {what} holdout dir (baselining the same benign you "
                     f"score/sweep is leakage). Use the holdout <-> holdout_cal pair.")
    if bool(args.baseline_dir) != bool(args.cal_baseline_dir):
        print("WARNING: baselines are on for one of {eval,cal} but not the other; "
              "the swept threshold may not transfer. Set both or neither.")

    print(f"Loading model bundles from {args.iso_model} / {args.xgb_model} ...")
    with open(args.iso_model, "rb") as f:
        iso_bundle = pickle.load(f)
    with open(args.xgb_model, "rb") as f:
        xgb_bundle = pickle.load(f)

    print("Preparing EVAL set ...")
    eval_set = prepare_set(args.holdout_dir, args.attack_dir, args.baseline_dir,
                           iso_bundle, xgb_bundle)
    same_as_eval = (str(args.cal_holdout_dir) == str(args.holdout_dir)
                    and str(args.cal_attack_dir) == str(args.attack_dir)
                    and str(args.cal_baseline_dir) == str(args.baseline_dir))
    if same_as_eval:
        print("CAL set == EVAL set (in-sample threshold sweep).")
        cal_set = eval_set
    else:
        print("Preparing CALIBRATION set (threshold swept here, frozen for eval) ...")
        cal_set = prepare_set(args.cal_holdout_dir, args.cal_attack_dir,
                              args.cal_baseline_dir, iso_bundle, xgb_bundle)

    print(f"eval: {eval_set['n']} events ({eval_set['n_attack']} attack) | "
          f"cal: {cal_set['n']} events ({cal_set['n_attack']} attack)")

    tag = " +base" if eval_set["base_events"] is not None else ""
    rows_spec = [
        ("B1. IF + risk (no graph)", "iso", "isolated"),
        (f"C1. IF + risk + graph{tag}", "iso", "graph"),
        (f"D1. IF + risk + oracle graph{tag}", "iso", "oracle"),
        ("B2. XGBoost + risk (no graph)", "xgb", "isolated"),
        (f"C2. XGBoost + risk + graph{tag}", "xgb", "graph"),
        (f"D2. XGBoost + risk + oracle graph{tag}", "xgb", "oracle"),
    ]
    rows = []
    for name, signal, mode in rows_spec:
        print(f"Running {name} ...")
        eval_res = run_row(eval_set, signal, mode)
        cal_res = eval_res if same_as_eval else run_row(cal_set, signal, mode)
        rows.append(metrics_report(name, eval_set["y"], eval_res,
                                   cal_set["y"], cal_res, metric=args.metric))

    print(f"\n{'=' * 108}")
    print("threshold = swept on the CAL set (max-%s), frozen, applied to EVAL." % args.metric)
    header = (f"{'row':<34}{'thr':>8}{'TP':>6}{'FP':>6}{'TN':>6}{'FN':>6}"
              f"{'prec':>8}{'recall':>8}{'f1':>8}{'fpr':>8}{'roc_auc':>9}{'pr_auc':>8}")
    print(header)
    for row in rows:
        print(f"{row['name']:<34}{row['calibrated_threshold']:>8.4f}{row['tp']:>6}{row['fp']:>6}"
              f"{row['tn']:>6}{row['fn']:>6}{row['precision']:>8.4f}"
              f"{row['recall']:>8.4f}{row['f1']:>8.4f}{row['fpr']:>8.4f}{row['roc_auc']:>9.4f}{row['pr_auc']:>8.4f}")
    print("=" * 108)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("row,calibrated_threshold,cal_f1,tp,fp,tn,fn,accuracy,precision,recall,f1,fpr,roc_auc,pr_auc\n")
        for row in rows:
            f.write(f"{row['name']},{row['calibrated_threshold']},{row['cal_f1']},{row['tp']},"
                     f"{row['fp']},{row['tn']},{row['fn']},{row['accuracy']},{row['precision']},"
                     f"{row['recall']},{row['f1']},{row['fpr']},{row['roc_auc']},{row['pr_auc']}\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
