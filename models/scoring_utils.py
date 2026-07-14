"""
scoring_utils.py — Shared helpers for scoring labeled data through a trained
Isolation Forest bundle, with CORRECT row alignment.

IMPORTANT: MLFeatureExtractor.extract_features() sorts internally by global
timestamp and resets the index (feature_extractor.py:
`df.sort_values('timestamp').reset_index(drop=True)`) -- so the row order of
its output X does NOT match the order events were passed in, especially when
loading multiple cloud files that are each individually sorted but not
globally interleaved (confirmed empirically: a 15-row cloud-grouped input
came back from extract_features in a completely different, timestamp-
interleaved order). Zipping a separately-built y_true/metadata array against
X by raw list position is WRONG and silently scores against misaligned
labels. Every function here carries an explicit __eval_row_id__ through
extract_features and uses it to realign labels/metadata back to the
ORIGINAL input order afterward -- this is the only correct way to do it.
Do not zip by position without going through featurize_and_align().
"""
import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from app.parser_normalizer.src.pipeline import ParserPipeline


def sweep_threshold(scores, y_true, metric="f1", precision_floor=0.9):
    """Pick an operating threshold by sweeping candidate cut points against
    LABELED ground truth, instead of a benign quantile.

    This is the single code path for threshold selection in this repo — both the
    Isolation Forest's decision_function threshold (validate_and_pick_threshold)
    and the risk-score alert_threshold (evaluate_full_pipeline) go through it, so
    the "swept against labels, not a benign quantile" lesson lives in one place
    (same discipline as featurize_and_align). A benign-quantile threshold (e.g.
    p99) is set by a handful of benign outliers and can land above every attack
    score even when the ranking is good — see models/README.md's threshold
    fragility finding.

    metric="f1"              -> the max-F1 cut point.
    metric="precision_floor" -> the highest-recall cut point whose precision is
                                >= precision_floor (falls back to the max-precision
                                point if the floor is unreachable).

    Returns a dict: threshold, precision, recall, f1, pr_auc, plus the full sweep
    arrays (thresholds / precision_curve / recall_curve / f1_curve) for plotting.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if len(set(y_true.tolist())) < 2:
        raise ValueError("sweep_threshold needs both classes present in y_true")

    pr_auc = float(average_precision_score(y_true, scores))
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds (final point has no
    # threshold); drop it so all three align.
    p, r = precision[:-1], recall[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)

    if metric == "precision_floor":
        ok = p >= precision_floor
        best = int(np.argmax(np.where(ok, r, -1.0))) if ok.any() else int(np.argmax(p))
    else:
        best = int(np.argmax(f1))

    return {
        "threshold": float(thresholds[best]),
        "precision": float(p[best]),
        "recall": float(r[best]),
        "f1": float(f1[best]),
        "pr_auc": pr_auc,
        "thresholds": thresholds,
        "precision_curve": p,
        "recall_curve": r,
        "f1_curve": f1,
    }


def discover_paths(base_dir: str, kind: str) -> dict:
    """Resolve the per-cloud JSON files in a split directory.

    kind="benign" matches `{cloud}_benign*.json` so it works for BOTH the
    `holdout` naming (`aws_benign_holdout.json`) and the Phase 0 `cal` naming
    (`aws_benign_cal.json`) without the caller hard-coding a suffix.
    kind="attack" matches `{cloud}_attack.json`. Errors if a cloud is missing or
    ambiguous, so a typo'd directory fails loudly rather than silently scoring a
    partial set.
    """
    pat = {"benign": "{c}_benign*.json", "attack": "{c}_attack.json"}[kind]
    out = {}
    for cloud in ("AWS", "AZURE", "GCP"):
        matches = sorted(glob.glob(os.path.join(base_dir, pat.format(c=cloud.lower()))))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {kind} file for {cloud} in {base_dir}, "
                f"found {len(matches)}: {matches}")
        out[cloud] = matches[0]
    return out


def ground_truth_identity(raw_log: dict, source_cloud: str):
    """Recover (event_id, gt_principal) straight from a raw provider record.

    gt_principal is the CANONICAL cross-cloud actor: the same human/service
    appears under a DIFFERENT normalized user_id per cloud (AWS arn vs Azure UPN
    vs GCP principalEmail), but the emitters embed the same bare actor name in
    each cloud's identity field. Canonicalizing all three back to that bare name
    gives a single ground-truth identity key that is correct ACROSS clouds — the
    thing graph-engine stitching is trying to reconstruct, so it's exactly what
    a merge's correctness (Phase 1b) and an oracle stitcher (Phase 1c) need.

    Note this is ground truth read from the generator's own identity fields, not
    an inference — the actor the emitter wrote is authoritative.
    """
    c = str(source_cloud or "").upper()
    if c == "AWS":
        ui = raw_log.get("userIdentity", {}) or {}
        name = ui.get("userName")
        if not name:
            arn = str(ui.get("arn", ""))
            name = arn.split("/")[-1] if "/" in arn else (arn or ui.get("principalId"))
        return raw_log.get("eventID"), name
    if c in ("AZURE", "ENTRA-ID"):
        props = raw_log.get("properties", {}) or {}
        upn = str(props.get("userPrincipalName") or props.get("servicePrincipalName") or "")
        return props.get("id"), (upn.split("@")[0] if upn else None)
    if c == "GCP":
        pe = str((raw_log.get("protoPayload", {}) or {})
                 .get("authenticationInfo", {}).get("principalEmail", ""))
        return raw_log.get("insertId"), (pe.split("@")[0] if pe else None)
    return None, None


def load_normalize_and_label(paths: dict):
    """Returns (unified_dicts, y_true, meta), all in the SAME original order.

    y_true[i] = 1 iff unified[i] is a genuine attack event
    (threat_category != "Normal"), 0 otherwise -- this intentionally
    excludes the generator's "mild" benign-oddity flag (generate_benign.py
    sets anomaly_flag=True / threat_category="Normal" for a small honest
    fraction of off-hours activity; that's an unusual-but-legitimate
    oddity, not an actual attack).

    meta[i] = dict with identifying fields (timestamp, source_cloud,
    user_id, action, threat_category, severity_score) for reporting which
    specific events were flagged.

    Each unified dict also carries a unique "__eval_row_id__" so
    featurize_and_align() can realign after extract_features() reorders
    rows internally.
    """
    pipeline = ParserPipeline()
    unified, y_true, meta = [], [], []
    row_id = 0
    for source_cloud, path in paths.items():
        with open(path, "r") as f:
            raw_logs = json.load(f)
        for raw_log in raw_logs:
            normalized = pipeline.process_log(raw_log, source_cloud=source_cloud)
            if normalized is None:
                continue
            labels = raw_log.get("ml_labels", {})
            category = labels.get("threat_category", "Normal")
            d = normalized.model_dump(exclude={"raw_log"})
            d["__eval_row_id__"] = row_id
            event_id, gt_principal = ground_truth_identity(raw_log, source_cloud)
            unified.append(d)
            y_true.append(int(category not in ("Normal", "", None)))
            meta.append({
                "timestamp": str(d.get("timestamp")),
                "source_cloud": d.get("source_cloud"),
                "user_id": d.get("user_id"),
                "action": d.get("action"),
                "threat_category": category,
                "severity_score": labels.get("severity_score"),
                # Phase 1: ground-truth join key + canonical cross-cloud actor,
                # for merge-purity auditing (1b) and the oracle stitcher (1c).
                "event_id": event_id,
                "gt_principal": gt_principal,
            })
            row_id += 1
    return unified, np.array(y_true), meta


def featurize_and_align(unified, extractor, feature_columns=None, is_training=False, **extract_kwargs):
    """Runs extract_features(), then realigns its output back to the
    ORIGINAL `unified` order using __eval_row_id__ (see module docstring for
    why this is necessary -- extract_features sorts by timestamp internally
    and resets the index).

    ALWAYS go through this function (never call extractor.extract_features()
    directly) on anything built via load_normalize_and_label() -- for
    TRAINING too, not just scoring/validation. Calling extract_features()
    directly and then zipping its output against a separately-built y array
    by position is exactly the bug this module exists to prevent, and it can
    silently reappear for training data the same way it did for validation
    data if this function is bypassed (confirmed: it did, in an early
    version of train_xgboost.py -- Normal, the largest class, got 0.00
    recall because X_train's rows and y_train were never realigned).

    feature_columns: pass the fitted training columns to reindex against
    (for scoring/test data, with is_training=False, reusing already-fitted
    label encoders). Leave None for training (is_training=True) -- there's
    no prior column list to reindex against yet; use list(X.columns) on the
    result as the feature_columns for subsequent featurize_and_align calls
    on validation/test data.

    Returns X (a DataFrame) such that X.iloc[i] corresponds to unified[i] /
    y_true[i] / meta[i] for whatever arrays the caller built alongside
    `unified` via load_normalize_and_label(). Callers can zip X's rows
    against those arrays directly by position after calling this.
    """
    X_raw, _ = extractor.extract_features(unified, is_training=is_training, **extract_kwargs)
    row_ids = X_raw["__eval_row_id__"].to_numpy()
    # row_ids[j] = original index of the event now at X_raw position j.
    # argsort(row_ids) gives the positions in ORIGINAL-index order, i.e.
    # the permutation that undoes extract_features' internal timestamp sort.
    order = np.argsort(row_ids)
    X = X_raw.iloc[order].reset_index(drop=True)
    X = X.drop(columns="__eval_row_id__")
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)  # defensive column-order guard
    return X
