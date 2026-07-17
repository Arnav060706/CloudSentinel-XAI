"""
evaluate_campaign_recall.py -- campaign-level (kill-chain-level) detection.

WHY THIS EXISTS
---------------
Event-level recall answers "what fraction of attack STEPS did we flag?".
Operationally, and for the paper's actual claim, the question that matters is
"did we catch the INTRUSION?" -- for each attack campaign, did we flag at
least ONE of its steps? A campaign whose 3rd of 17 steps is flagged IS
detected; event-level recall counts the other 16 steps as misses and so can
badly understate real detection capability.

This runs on the CALIBRATION set (holdout_cal + attacks_cal) and REFUSES to
touch the frozen attacks_slow split without an explicit override. See
"0. The ordering constraint" and models/README.md: all selection/exploration
happens on cal; attacks_slow is consumed once, post-freeze.

CAMPAIGN RECALL IS GAMEABLE -- READ THIS BEFORE REPORTING IT
------------------------------------------------------------
A campaign counts as caught if ANY ONE of its steps is flagged, so LOWERING the
threshold inflates campaign recall much faster than event recall. Measured here:
row B1 at its max-F1 threshold reaches campaign recall 1.000 (32/32) at event
precision 0.116 -- i.e. flagging ~three quarters of all traffic "detects" every
campaign trivially. Campaign recall MUST therefore always be reported next to
precision (or an alert budget), never alone.

This also means rows compared at their OWN max-F1 thresholds are not
comparable: B2 reaches 31/32 campaigns at precision 0.192 while C2 reaches
15/32 at precision 0.562. `--precision-targets` fixes this by re-deriving each
row's operating point at a COMMON event-precision floor, so campaign recall is
compared at equal alert quality. That is the honest cross-row comparison.

CAMPAIGN IDENTITY -- THE NON-OBVIOUS PART
-----------------------------------------
generate_attacks.py runs

    for _ in range(repeats):          # 8
        for scen in ALL_SCENARIOS:    # 4
            actor = rng.choice(pools[scen["victim_role"]])   # fresh random victim

so there are exactly repeats * len(ALL_SCENARIOS) = 32 campaigns. But it emits
only `scenario` and `actor` into attack_labels.csv -- there is NO campaign or
repeat id. The random draw sometimes picks the SAME victim twice for the SAME
scenario, so (scenario, actor) is NOT a valid campaign key. Measured:

    attacks_cal : 31 distinct (scenario,actor) pairs vs 32 real campaigns
                  -- alice.chen has 34 = 2x17 cross_cloud_apt events
    attacks_slow: 29 distinct pairs vs 32 real campaigns
                  -- yuki.park 34 = 2x17, svc-ci-pipeline 15 = 3x5

Grouping by (scenario, actor) would MERGE those campaigns, and a merged campaign
needs only one detected event to count as caught -- INFLATING campaign recall.
Campaigns are therefore also split on a time gap:
  - within a campaign, "slow" pacing puts <= ~1.5h between steps (largest
    observed intra gap 1.49h)
  - distinct campaigns of the same (scenario, actor) are >= 34h apart
    (observed 34.1h for alice.chen, ~142h for svc-ci-pipeline)
--gap-hours 6 sits ~4x above the largest intra gap and ~5.7x below the smallest
inter gap. The clustering is VALIDATED, not trusted: the true count is known
from the generator and this script FAILS LOUDLY if clustering misses it.

THRESHOLD CAVEAT
----------------
Thresholds are swept on the labeled cal set and evaluated on that same cal set,
so operating points are IN-SAMPLE. Acceptable here because the purpose is
structural (is campaign recall materially higher than event recall, and does the
graph layer help at matched precision?), not to produce a headline number. The
reportable figure comes from the single post-freeze pass on attacks_slow.

Run:
    python models/evaluate_campaign_recall.py
"""
import argparse
import csv
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_full_pipeline import prepare_set, run_row
from scoring_utils import sweep_threshold

BENIGN_NOISE = "benign_noise"

ROWS = [
    ("B1. IF, no graph",       "iso", "isolated"),
    ("C1. IF + graph",         "iso", "graph"),
    ("D1. IF + oracle graph",  "iso", "oracle"),
    ("B2. XGB, no graph",      "xgb", "isolated"),
    ("C2. XGB + graph",        "xgb", "graph"),
    ("D2. XGB + oracle graph", "xgb", "oracle"),
]


def build_campaign_map(attack_dir, gap_hours=6.0):
    """event_id -> campaign_id for every ATTACK event in attack_labels.csv.

    A campaign is a run of same-(scenario, actor) events with no internal gap
    larger than `gap_hours`. See the module docstring for why (scenario, actor)
    alone is insufficient and why this threshold is safe.
    """
    df = pd.read_csv(Path(attack_dir) / "attack_labels.csv", parse_dates=["timestamp"])
    atk = df[df.scenario != BENIGN_NOISE].copy()

    campaign_of = {}
    n = 0
    max_intra = 0.0
    min_inter = float("inf")

    for (scen, actor), g in atk.groupby(["scenario", "actor"], sort=True):
        g = g.sort_values("timestamp")
        gaps_h = g["timestamp"].diff().dt.total_seconds() / 3600.0
        is_break = (gaps_h > gap_hours).fillna(False)
        for k, sub in g.groupby(is_break.cumsum()):
            cid = f"{scen}|{actor}|{int(k)}"
            for eid in sub["event_id"]:
                campaign_of[eid] = cid
            n += 1
        intra = gaps_h[~is_break].dropna()
        inter = gaps_h[is_break].dropna()
        if len(intra):
            max_intra = max(max_intra, float(intra.max()))
        if len(inter):
            min_inter = min(min_inter, float(inter.min()))

    return campaign_of, n, {
        "max_intra_campaign_gap_h": round(max_intra, 2),
        "min_inter_campaign_gap_h": (None if min_inter == float("inf")
                                     else round(min_inter, 2)),
    }


def campaign_structs(event_ids, y, scores, campaign_of):
    """Per-campaign score structures.

    prepare_set() returns events in GLOBAL CHRONOLOGICAL order, so appending
    per-campaign preserves STEP order -- that's what makes first-detect-step
    meaningful.

    Returns (camp_scores, unmapped) where camp_scores maps
    campaign_id -> np.array of that campaign's attack-event scores in step order.
    """
    camp_scores = defaultdict(list)
    unmapped = 0
    for eid, yt, s in zip(event_ids, y, scores):
        if int(yt) != 1:
            continue
        cid = campaign_of.get(eid)
        if cid is None:
            unmapped += 1
            continue
        camp_scores[cid].append(float(s))
    return {c: np.asarray(v) for c, v in camp_scores.items()}, unmapped


def event_stats_at(scores, y, thr):
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "precision": prec, "recall": rec}


def campaign_stats_at(camp_scores, thr):
    """A campaign is caught iff any of its steps scores >= thr."""
    caught, first_steps, fracs = set(), [], []
    for cid, arr in camp_scores.items():
        hits = np.nonzero(arr >= thr)[0]
        if len(hits):
            caught.add(cid)
            k = int(hits[0]) + 1               # 1-based step index of first hit
            first_steps.append(k)
            fracs.append(k / len(arr))
    n = len(camp_scores)
    return {
        "caught": caught,
        "n_campaigns": n,
        "campaigns_detected": len(caught),
        "campaign_recall": (len(caught) / n) if n else 0.0,
        "median_first_detect_step": (statistics.median(first_steps)
                                     if first_steps else None),
        "median_frac_through": (round(statistics.median(fracs), 3)
                                if fracs else None),
    }


def op_at_precision(scores, y, camp_scores, target):
    """Operating point maximizing CAMPAIGN recall subject to event precision
    >= target. Precision is only near-monotonic in threshold, so scan every
    candidate rather than assuming a crossing point. Returns None if the
    precision floor is unreachable for this row.
    """
    best = None
    for t in np.unique(scores):
        es = event_stats_at(scores, y, t)
        if (es["tp"] + es["fp"]) == 0 or es["precision"] < target:
            continue
        cs = campaign_stats_at(camp_scores, t)
        if best is None or cs["campaign_recall"] > best["campaign_recall"]:
            best = {"threshold": float(t), **es, **cs}
    return best


def per_scenario(caught, camp_scores):
    d = defaultdict(lambda: {"caught": 0, "total": 0})
    for cid in camp_scores:
        scen = cid.split("|")[0]
        d[scen]["total"] += 1
        if cid in caught:
            d[scen]["caught"] += 1
    return dict(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-model", default="models/iso_forest.pkl")
    ap.add_argument("--xgb-model", default="models/xgboost_classifier.pkl")
    ap.add_argument("--holdout-dir", default="Datasets/holdout_cal")
    ap.add_argument("--attack-dir", default="Datasets/attacks_cal")
    ap.add_argument("--baseline-dir", default="Datasets/holdout",
                    help="disjoint benign warmup for per-identity baselines")
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--expect-campaigns", type=int, default=32,
                    help="generator ground truth: --repeats x len(ALL_SCENARIOS). "
                         "0 disables the check.")
    ap.add_argument("--precision-targets", default="0.3,0.5,0.7",
                    help="common event-precision floors for the matched comparison")
    ap.add_argument("--frozen-run", action="store_true",
                    help="REQUIRED to point this at Datasets/attacks_slow. Only for "
                         "the single post-freeze pass.")
    ap.add_argument("--out", default="models/campaign_recall_cal.csv")
    ap.add_argument("--out-matched", default="models/campaign_recall_matched_cal.csv")
    args = ap.parse_args()

    if "attacks_fast" in str(args.attack_dir):
        raise SystemExit("Refusing --attack-dir Datasets/attacks_fast: XGBoost's "
                         "TRAINING data; any XGB row measured on it is leakage.")
    if "attacks_slow" in str(args.attack_dir) and not args.frozen_run:
        raise SystemExit(
            "Refusing to touch Datasets/attacks_slow without --frozen-run.\n"
            "It is the frozen final test split, consumed ONCE after the config is "
            "locked. This is CAL-set work: use --attack-dir Datasets/attacks_cal.")
    if Path(args.baseline_dir).name == Path(args.holdout_dir).name:
        raise SystemExit("--baseline-dir must be DISJOINT from --holdout-dir.")

    targets = [float(t) for t in args.precision_targets.split(",")]

    with open(args.iso_model, "rb") as f:
        iso_bundle = pickle.load(f)
    with open(args.xgb_model, "rb") as f:
        xgb_bundle = pickle.load(f)

    campaign_of, n_camp, diag = build_campaign_map(args.attack_dir, args.gap_hours)
    print(f"campaign clustering on {args.attack_dir} (gap > {args.gap_hours}h = new)")
    print(f"  campaigns found ........ {n_camp}")
    print(f"  max intra-campaign gap . {diag['max_intra_campaign_gap_h']}h")
    print(f"  min inter-campaign gap . {diag['min_inter_campaign_gap_h']}h")
    if args.expect_campaigns and n_camp != args.expect_campaigns:
        raise SystemExit(
            f"\nFAIL: clustering recovered {n_camp}, generator ground truth says "
            f"{args.expect_campaigns}. The campaign key is wrong -- every number "
            f"below would be invalid. Check --gap-hours / --expect-campaigns.")
    print(f"  VALIDATED against generator ground truth ({args.expect_campaigns}).\n")

    setd = prepare_set(args.holdout_dir, args.attack_dir, args.baseline_dir,
                       iso_bundle, xgb_bundle)
    y = np.asarray(setd["y"])
    eids = setd["event_id"]
    print(f"cal set: {setd['n']} events, {setd['n_attack']} attack "
          f"({setd['n_attack']/setd['n']:.2%} base rate)\n")

    rows_out, matched_out = [], []
    caught_by_row = {}

    for name, signal, mode in ROWS:
        results = run_row(setd, signal, mode)
        scores = np.array([r["scaled_score"] for r in results])
        camp_scores, unmapped = campaign_structs(eids, y, scores, campaign_of)
        if unmapped:
            raise SystemExit(f"FAIL: {unmapped} attack events have no campaign row "
                             f"in attack_labels.csv -- join key broken.")

        # --- row's own max-F1 operating point (NOT comparable across rows) ---
        thr = sweep_threshold(scores, y, metric="f1")["threshold"]
        es = event_stats_at(scores, y, thr)
        cs = campaign_stats_at(camp_scores, thr)
        caught_by_row[name] = cs["caught"]
        lift = (cs["campaign_recall"] / es["recall"]) if es["recall"] else float("nan")

        rows_out.append({
            "row": name, "threshold": round(thr, 4),
            "event_recall": round(es["recall"], 4),
            "event_precision": round(es["precision"], 4),
            "fp": es["fp"],
            "campaigns_detected": cs["campaigns_detected"],
            "n_campaigns": cs["n_campaigns"],
            "campaign_recall": round(cs["campaign_recall"], 4),
            "campaign_vs_event_recall_x": round(lift, 2),
            "median_first_detect_step": cs["median_first_detect_step"],
            "median_frac_through_campaign": cs["median_frac_through"],
        })

        print(f"--- {name} (max-F1 thr {thr:.4f}) ---")
        print(f"  event    recall {es['recall']:.3f}  precision {es['precision']:.3f}  "
              f"FP {es['fp']}")
        print(f"  CAMPAIGN recall {cs['campaign_recall']:.3f} "
              f"({cs['campaigns_detected']}/{cs['n_campaigns']}) = {lift:.2f}x event")
        for scen, d in sorted(per_scenario(cs["caught"], camp_scores).items()):
            print(f"      {scen:<36} {d['caught']}/{d['total']}")

        # --- matched-precision operating points (comparable across rows) ------
        for tgt in targets:
            op = op_at_precision(scores, y, camp_scores, tgt)
            if op is None:
                print(f"  @precision>={tgt:.2f}: UNREACHABLE for this row")
                matched_out.append({"row": name, "precision_target": tgt,
                                    "reachable": False, "threshold": None,
                                    "event_precision": None, "event_recall": None,
                                    "fp": None, "campaigns_detected": None,
                                    "campaign_recall": None,
                                    "median_first_detect_step": None})
                continue
            print(f"  @precision>={tgt:.2f}: campaign recall "
                  f"{op['campaign_recall']:.3f} ({op['campaigns_detected']}/"
                  f"{op['n_campaigns']})  event recall {op['recall']:.3f}  "
                  f"FP {op['fp']}  thr {op['threshold']:.4f}")
            matched_out.append({
                "row": name, "precision_target": tgt, "reachable": True,
                "threshold": round(op["threshold"], 4),
                "event_precision": round(op["precision"], 4),
                "event_recall": round(op["recall"], 4),
                "fp": op["fp"],
                "campaigns_detected": op["campaigns_detected"],
                "campaign_recall": round(op["campaign_recall"], 4),
                "median_first_detect_step": op["median_first_detect_step"],
            })
        print()

    # --- identity check: are C1/C2 (and D1/D2) catching the SAME campaigns? ---
    print("=" * 66)
    print("CAUGHT-CAMPAIGN SET COMPARISON (at each row's max-F1 threshold)")
    print("Identical sets across two different signals would be structural --")
    print("the graph decides which campaigns light up -- but identical numbers")
    print("are also a classic bug signature, so compare the SETS, not counts.")
    print("=" * 66)
    for a, b in [("C1. IF + graph", "C2. XGB + graph"),
                 ("D1. IF + oracle graph", "D2. XGB + oracle graph"),
                 ("B1. IF, no graph", "B2. XGB, no graph")]:
        sa, sb = caught_by_row[a], caught_by_row[b]
        inter, union = sa & sb, sa | sb
        jac = len(inter) / len(union) if union else 1.0
        verdict = "IDENTICAL" if sa == sb else "DIFFERENT"
        print(f"\n{a}  vs  {b}")
        print(f"  |A|={len(sa)}  |B|={len(sb)}  shared={len(inter)}  "
              f"Jaccard={jac:.3f}  -> {verdict}")
        if sa != sb:
            for cid in sorted(sa - sb):
                print(f"    only in A: {cid}")
            for cid in sorted(sb - sa):
                print(f"    only in B: {cid}")

    for path, data in [(args.out, rows_out), (args.out_matched, matched_out)]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"\nSaved -> {p}")

    print("\nNOTE: cal set, in-sample thresholds. Structural question only; the "
          "reportable number comes from the single post-freeze pass.")


if __name__ == "__main__":
    main()
