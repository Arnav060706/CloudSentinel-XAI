"""
audit_entity_purity.py — Phase 1b: score the graph engine's identity stitching
against GROUND TRUTH, to root-cause why the graph layer hurts detection on
slow-paced data (models/README.md "Known gaps" #5).

What it does
------------
Replays the calibration set (holdout_cal + attacks_cal, per Global rule 2 — the
final test set attacks_slow is NOT used for this diagnosis) through
MultiCloudGraphEngine in chronological order, with the Phase-1a merge audit log
enabled, then:

  * Per entity: how many DISTINCT ground-truth actors it absorbed, how many
    attack vs benign events, and a purity flag:
      - "pure"            : one ground-truth actor
      - "mixed-principal" : >1 actor, all same label (all benign or all attack)
      - "mixed-label"     : >1 actor spanning both benign and attack events
    Only mixed entities can distort the per-entity risk signal; mixed-label ones
    (benign stitched onto attack, or two unrelated campaigns) are the ones that
    plausibly explain C < B.

  * Per Tier-2 fuzzy merge: was it CORRECT (re-stitched the SAME actor across
    clouds) or WRONG (united two different actors on fuzzy similarity)? For every
    WRONG merge it prints the per-signal similarity breakdown (which of UA /
    proxy / IP / type over-credited), computed via
    MultiCloudGraphEngine.explain_fuzzy_similarity against the entity anchor.

Outputs: models/entity_purity.csv, models/merge_audit.csv, and a printed summary.

Usage:
  python models/audit_entity_purity.py \
      --holdout-dir Datasets/holdout_cal --attack-dir Datasets/attacks_cal
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.services.graph_engine as graph_engine_module
from app.services.graph_engine import MultiCloudGraphEngine
from scoring_utils import load_normalize_and_label, discover_paths


def _epoch(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def build_events(unified, y_true, meta):
    """One event dict per record, carrying the fields the graph engine needs
    (principal = normalized user_id, matching evaluate_full_pipeline.py) plus
    the ground-truth join fields from meta (event_id, gt_principal, label)."""
    events = []
    for u, y, m in zip(unified, y_true, meta):
        e = dict(u)
        e["principal"] = u.get("user_id", "")
        e["event_id"] = m.get("event_id")
        e["gt_principal"] = m.get("gt_principal")
        e["is_attack"] = int(y)
        events.append(e)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-dir", default="Datasets/holdout_cal")
    ap.add_argument("--attack-dir", default="Datasets/attacks_cal",
                    help="MUST NOT be attacks_fast (XGBoost training data). "
                         "Defaults to the cal split per Global rule 2.")
    ap.add_argument("--out-purity", default="models/entity_purity.csv")
    ap.add_argument("--out-audit", default="models/merge_audit.csv")
    args = ap.parse_args()

    if "attacks_fast" in str(args.attack_dir):
        sys.exit("Refusing to audit on attacks_fast (XGBoost training data).")

    benign_paths = discover_paths(args.holdout_dir, "benign")
    attack_paths = discover_paths(args.attack_dir, "attack")
    print(f"benign : {args.holdout_dir}")
    print(f"attack : {args.attack_dir}")

    unified_b, y_b, meta_b = load_normalize_and_label(benign_paths)
    unified_a, y_a, meta_a = load_normalize_and_label(attack_paths)
    unified = list(unified_b) + list(unified_a)
    import numpy as np
    y_true = np.concatenate([y_b, y_a])
    meta = list(meta_b) + list(meta_a)

    events = build_events(unified, y_true, meta)
    events.sort(key=lambda e: _epoch(e["timestamp"]))  # chronological (stateful engine)

    n_attack = int(sum(e["is_attack"] for e in events))
    print(f"{len(events)} events ({n_attack} attack, {len(events) - n_attack} benign)")

    # --- Replay through the graph engine with the merge audit log enabled ----
    audit_log = []
    graph = MultiCloudGraphEngine(merge_audit_log=audit_log)

    class _Clock:
        now = 0.0
    clock = _Clock()
    real_time_time = graph_engine_module.time.time
    graph_engine_module.time.time = lambda: clock.now
    try:
        event_entity = []  # (event, entity_id)
        for e in events:
            clock.now = _epoch(e["timestamp"])
            entity_id, _active, _new, _method, _clouds = graph.process_event(e)
            event_entity.append((e, entity_id))
    finally:
        graph_engine_module.time.time = real_time_time

    # --- Per-entity purity ---------------------------------------------------
    ent_gt = defaultdict(set)       # entity_id -> set(gt_principal)
    ent_attack = defaultdict(int)
    ent_benign = defaultdict(int)
    ent_events = defaultdict(int)
    ent_clouds = defaultdict(set)
    for e, eid in event_entity:
        ent_gt[eid].add(e.get("gt_principal"))
        ent_events[eid] += 1
        ent_clouds[eid].add(str(e.get("source_cloud")))
        if e["is_attack"]:
            ent_attack[eid] += 1
        else:
            ent_benign[eid] += 1

    def purity_flag(eid):
        if len(ent_gt[eid]) <= 1:
            return "pure"
        if ent_attack[eid] > 0 and ent_benign[eid] > 0:
            return "mixed-label"
        return "mixed-principal"

    purity_rows = []
    for eid in ent_events:
        purity_rows.append({
            "entity_id": eid,
            "n_events": ent_events[eid],
            "n_distinct_gt_principals": len(ent_gt[eid]),
            "n_attack_events": ent_attack[eid],
            "n_benign_events": ent_benign[eid],
            "n_clouds": len(ent_clouds[eid]),
            "purity": purity_flag(eid),
            "gt_principals": "|".join(sorted(str(g) for g in ent_gt[eid])),
        })
    purity_rows.sort(key=lambda r: (r["purity"] != "mixed-label",
                                    r["purity"] != "mixed-principal",
                                    -r["n_events"]))

    # --- Per-fuzzy-merge correctness ----------------------------------------
    # Reconstruct each entity's accumulated ground-truth set in processing order;
    # a fuzzy_merge is WRONG if it introduces a gt actor the entity didn't
    # already contain (i.e. it stitched a genuinely different identity).
    eid_to_gt = {e["event_id"]: e.get("gt_principal") for e in events}
    eid_to_event = {e["event_id"]: e for e in events}
    entity_gt_running = defaultdict(set)
    fuzzy_total = fuzzy_wrong = 0
    wrong_merges = []
    for rec in audit_log:
        eid = rec["entity_id"]
        gt = eid_to_gt.get(rec["event_id"])
        if rec["method"] == "fuzzy_merge":
            fuzzy_total += 1
            prior = set(entity_gt_running[eid])
            if gt not in prior and prior:  # merged a new, different actor in
                fuzzy_wrong += 1
                anchor = graph._entity_anchors.get(eid, {})
                ev = eid_to_event.get(rec["event_id"], {})
                breakdown = graph.explain_fuzzy_similarity(ev, anchor)
                wrong_merges.append({
                    "event_id": rec["event_id"],
                    "entity_id": eid,
                    "merged_gt": gt,
                    "entity_gt_so_far": "|".join(sorted(str(g) for g in prior)),
                    "similarity": rec["similarity_score"],
                    "breakdown": breakdown,
                })
        entity_gt_running[eid].add(gt)

    # --- Write CSVs ----------------------------------------------------------
    Path(args.out_purity).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_purity, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(purity_rows[0].keys()))
        w.writeheader(); w.writerows(purity_rows)
    with open(args.out_audit, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "event_id", "tier", "method", "similarity_score", "entity_id",
            "event_principal", "entity_principals_so_far"])
        w.writeheader()
        for rec in audit_log:
            r = dict(rec)
            r["entity_principals_so_far"] = "|".join(rec["entity_principals_so_far"])
            w.writerow(r)

    # --- Printed summary -----------------------------------------------------
    n_entities = len(ent_events)
    n_pure = sum(1 for r in purity_rows if r["purity"] == "pure")
    n_mixed_p = sum(1 for r in purity_rows if r["purity"] == "mixed-principal")
    n_mixed_l = sum(1 for r in purity_rows if r["purity"] == "mixed-label")
    method_counts = defaultdict(int)
    for rec in audit_log:
        method_counts[rec["method"]] += 1

    print("\n" + "=" * 72)
    print(f"Entities: {n_entities} from "
          f"{len({e.get('gt_principal') for e in events})} distinct ground-truth actors")
    print(f"  pure={n_pure}  mixed-principal={n_mixed_p}  mixed-label={n_mixed_l}")
    print(f"Resolution methods: {dict(method_counts)}")
    print(f"Fuzzy merges: {fuzzy_total} total, {fuzzy_wrong} WRONG "
          f"(united different actors)")
    print("=" * 72)
    if wrong_merges:
        print("\nWRONG fuzzy merges (per-signal similarity vs entity anchor):")
        # which signal is most often responsible?
        signal_blame = defaultdict(float)
        for wm in wrong_merges:
            b = wm["breakdown"]
            print(f"  event {wm['event_id']} gt='{wm['merged_gt']}' merged into "
                  f"entity holding '{wm['entity_gt_so_far']}' @sim={wm['similarity']}")
            print(f"      ua={b['ua']} proxy={b['proxy']} ip={b['ip']} "
                  f"type={b['type']} total={b['total']}")
            for k in ("ua", "proxy", "ip", "type"):
                signal_blame[k] += b[k]
        print(f"\n  Total similarity credit by signal across wrong merges: "
              f"{ {k: round(v, 3) for k, v in signal_blame.items()} }")
    print(f"\nWrote {args.out_purity} and {args.out_audit}")


if __name__ == "__main__":
    main()
