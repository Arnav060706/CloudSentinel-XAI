"""
generate_attacks.py — MITRE ATT&CK cross-cloud attacks (leakage-safe)
=====================================================================

Renders the kill chains in attack_scenarios.py into schema-correct logs,
interleaved with benign noise (realistic normal+anomalous mix).

ANTI-LEAKAGE DESIGN:
  * Each scenario's victim is a randomly-drawn LEGITIMATE identity of the
    required role (from the same population that appears benign). Across seeds,
    every user is sometimes benign and sometimes a victim — so no username
    correlates with the label.
  * Victims are drawn from BOTH held-out and train users, so the test set
    contains attacks on principals the model never saw benign — the
    generalization experiment.
  * The ONLY attack-exclusive identities are accounts the attacker *creates*
    mid-chain (e.g. bd-svc-01); that is realistic and their creation event is
    itself part of the signal, not a leak.

Controls: --pace {fast,slow,mixed}   --noise-ratio   --seed
Usage: python generate_attacks.py --seed 7 --pace mixed --noise-ratio 0.5 --out ./out
"""
from __future__ import annotations
import argparse, json, os, csv, datetime as dt
from env_profile import (Environment, ml_label, BROWSER_UAS, SDK_UAS_GCP,
                         SCRIPT_UAS, SUSPICIOUS_UAS, BENIGN_GEO, FOREIGN_GEO,
                         CORP_DOMAIN, GCP_PROJECT, AWS_ACCOUNTS)
import emitters as em
from attack_scenarios import ALL_SCENARIOS

INFRA_GEO = {"tor": FOREIGN_GEO, "foreign": FOREIGN_GEO, "hosting": FOREIGN_GEO,
             "home": BENIGN_GEO, "office": BENIGN_GEO}
PACE = {"fast": (5, 90), "slow": (600, 5400)}

def _gap(pace, rng):
    band = rng.choice(["fast","slow"]) if pace == "mixed" else pace
    return rng.randint(*PACE[band])

def _emit(env, s, actor, is_service, ts, rows, aws, az, gcp, scen):
    rng = env.rng; infra = s["infra"]; ip = env.pick_ip(infra)
    lbl = ml_label(True, s["category"], s["severity"]); cloud = s["cloud"]
    upn = env.upn(actor); display = actor.replace(".", " ").replace("-", " ").title()
    created = s.get("created")
    eid = None
    if cloud == "aws":
        ua = rng.choice(SUSPICIOUS_UAS) if infra in ("tor","hosting","foreign") else rng.choice(BROWSER_UAS)
        # Real CloudTrail CreateUser/CreateAccessKey events carry the target
        # identity in requestParameters. Without this, the backdoor account
        # never appears anywhere in the log content (only in the note/label),
        # so nothing in the raw event distinguishes the identity being created.
        request_params = {"userName": created} if created else None
        rec = em.aws_event(ts=ts, event_name=s["action"], user_name=actor,
            account_id=AWS_ACCOUNTS[0], ip=ip, ua=ua, success=s["success"], mfa=False,
            principal_type="AssumedRole" if is_service else "IAMUser",
            error_code=None if s["success"] else "FailedAuthentication",
            read_only=s["action"].startswith(("List","Get")), labels=lbl,
            request_params=request_params)
        aws.append(rec); eid = rec["eventID"]
    elif cloud == "azure":
        rec = em.azure_event(ts=ts, operation=s["action"], upn=upn, display_name=display,
            ip=ip, ua=rng.choice(SUSPICIOUS_UAS), success=s["success"], mfa=False,
            country=rng.choice(INFRA_GEO.get(infra, FOREIGN_GEO)), compliant=False,
            managed=False, labels=lbl)
        az.append(rec); eid = rec["properties"]["id"]
    else:
        pe = env.gcp_principal(actor, is_service)
        ua = rng.choice(SCRIPT_UAS) if infra in ("tor","hosting","foreign") else rng.choice(SDK_UAS_GCP)
        sev = "ERROR" if not s["success"] else ("WARNING" if s["severity"]>=0.6 else "NOTICE")
        request = None
        if created:
            if "CreateServiceAccountKey" in s["action"]:
                request = {"name": f"projects/{GCP_PROJECT}/serviceAccounts/"
                                    f"{created}@{GCP_PROJECT}.iam.gserviceaccount.com"}
            elif "CreateServiceAccount" in s["action"]:
                request = {"accountId": created}
        rec = em.gcp_event(ts=ts, method=s["action"], principal_email=pe, ip=ip, ua=ua,
            success=s["success"], severity=sev, labels=lbl, request=request)
        gcp.append(rec); eid = rec["insertId"]
    rows.append({"event_id": eid, "scenario": scen, "actor": actor, "cloud": cloud,
        "action": s["action"], "timestamp": ts, "anomaly_flag": True,
        "threat_category": s["category"], "severity_score": s["severity"],
        "tactic": s["tactic"], "technique_id": s["technique_id"],
        "technique_name": s["technique_name"], "infra": infra, "note": s["note"],
        "created": created or ""})

def _noise(env, day0, rng, n, aws, az, gcp, rows):
    # benign events using RANDOM legitimate users (incl. victims) -> no name/label link
    from generate_benign import AWS_USER, AZ_USER, GCP_USER
    for _ in range(n):
        user = rng.choice(env.human_names)
        ts = (day0 + dt.timedelta(days=rng.randint(0,10), hours=rng.randint(8,18),
              minutes=rng.randint(0,59))).strftime("%Y-%m-%dT%H:%M:%SZ")
        cloud = rng.choice(["aws","azure","gcp"]); ip = env.pick_benign_ip()
        lbl = ml_label(False, "Normal", round(rng.uniform(0.02,0.14),2))
        if cloud == "aws":
            rec = em.aws_event(ts=ts, event_name=rng.choice(AWS_USER), user_name=user,
                account_id=AWS_ACCOUNTS[0], ip=ip, ua=rng.choice(BROWSER_UAS),
                success=True, mfa=True, read_only=True, labels=lbl); aws.append(rec); eid=rec["eventID"]
        elif cloud == "azure":
            rec = em.azure_event(ts=ts, operation="Sign-in activity", upn=env.upn(user),
                display_name=user.title(), ip=ip, ua=rng.choice(BROWSER_UAS), success=True,
                mfa=True, country=rng.choice(BENIGN_GEO), labels=lbl); az.append(rec); eid=rec["properties"]["id"]
        else:
            rec = em.gcp_event(ts=ts, method="GetIamPolicy", principal_email=env.upn(user),
                ip=ip, ua=rng.choice(SDK_UAS_GCP), success=True, labels=lbl); gcp.append(rec); eid=rec["insertId"]
        rows.append({"event_id": eid, "scenario": "benign_noise", "actor": user, "cloud": cloud,
            "action": rec.get("eventName") or "Sign-in activity", "timestamp": ts,
            "anomaly_flag": False, "threat_category": "Normal",
            "severity_score": lbl["severity_score"], "tactic": "", "technique_id": "",
            "technique_name": "", "infra": "benign", "note": ""})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="attack randomness")
    ap.add_argument("--env-seed", type=int, default=42, help="population identity (MUST match benign)")
    ap.add_argument("--pace", choices=["fast","slow","mixed"], default="mixed")
    ap.add_argument("--noise-ratio", type=float, default=0.5)
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=1,
                    help="times to run each scenario (each on a fresh random victim)")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--days", type=int, default=21,
                    help="length of the benign observation window (MUST match generate_benign.py --days) "
                         "so attack instances stay inside the range where benign data also exists")
    args = ap.parse_args()
    env = Environment(seed=args.env_seed, n_users=args.n_users)
    import random as _r; rng = _r.Random(args.seed); env.rng = rng
    em.reset_ids(); em.set_run_tag(f"atk-{args.pace}"); os.makedirs(args.out, exist_ok=True)

    # BENIGN_START must match generate_benign.py's day0 (2026-06-01). Attack chains
    # start a few days in (so there's benign history before them) and must all
    # finish before the observation window closes, or a model could "detect"
    # attacks by date alone rather than by behavior.
    BENIGN_START = dt.datetime(2026, 6, 1)
    START_OFFSET_DAYS = 3
    END_BUFFER_DAYS = 2
    day0 = BENIGN_START + dt.timedelta(days=START_OFFSET_DAYS, hours=2)

    aws, az, gcp, rows = [], [], [], []
    pools = {"human_admin": env.admin_names, "human_user": env.human_names,
             "service_account": env.service_names}
    n_instances = args.repeats * len(ALL_SCENARIOS)
    available_days = max(args.days - START_OFFSET_DAYS - END_BUFFER_DAYS, 1)
    step_days = available_days / max(n_instances, 1)
    slot = 0
    for _ in range(args.repeats):
        for scen in ALL_SCENARIOS:
            role = scen["victim_role"]
            actor = rng.choice(pools[role])                 # <-- random legit victim
            is_service = role == "service_account"
            t = day0 + dt.timedelta(days=slot*step_days, hours=rng.randint(0,6)); slot += 1
            # tag which split this victim belongs to (for the generalization experiment)
            victim_split = "holdout" if actor in env.holdout_users else "train"
            for s in scen["steps"]:
                _emit(env, s, actor, is_service, t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      rows, aws, az, gcp, scen["name"])
                rows[-1]["victim_split"] = victim_split
                t += dt.timedelta(seconds=_gap(args.pace, rng))

    n_atk = sum(1 for r in rows if r["anomaly_flag"])
    _noise(env, day0, rng, int(n_atk*args.noise_ratio), aws, az, gcp, rows)
    for r in rows:
        r.setdefault("victim_split", "")
        r.setdefault("created", "")

    for st in (aws, az, gcp):
        st.sort(key=lambda r: r.get("eventTime") or r.get("time") or r.get("timestamp"))
    json.dump(aws, open(f"{args.out}/aws_attack.json","w"), indent=2)
    json.dump(az,  open(f"{args.out}/azure_attack.json","w"), indent=2)
    json.dump(gcp, open(f"{args.out}/gcp_attack.json","w"), indent=2)
    with open(f"{args.out}/attack_labels.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    total = len(aws)+len(az)+len(gcp)
    techs = sorted(set(r["technique_id"] for r in rows if r["technique_id"]))
    holdout_atk = sum(1 for r in rows if r.get("victim_split")=="holdout")
    print(f"[attack] total={total} malicious={n_atk} ({100*n_atk/total:.1f}%) "
          f"| techniques={len(techs)} pace={args.pace} repeats={args.repeats} "
          f"| holdout-victim events={holdout_atk} -> {args.out}")

if __name__ == "__main__":
    main()