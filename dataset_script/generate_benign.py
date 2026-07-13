"""
generate_benign.py — Benign multi-cloud IAM activity (procedural population)
===========================================================================

Generates a realistic BENIGN baseline over a large seeded organization.
Held-out identities are EXCLUDED here (they appear only in the test/attack
split) so you can measure generalization to unseen principals.

Labels come from THIS script (anomaly_flag=False for benign, plus a small
honest fraction of low-severity benign oddities). No identity is inherently
good or bad — the same users are later drawn as attack victims.

Usage: python generate_benign.py --days 14 --seed 42 --n-users 300 --out ./out
"""
from __future__ import annotations
import argparse, json, os, datetime as dt
from env_profile import (Environment, ml_label, BROWSER_UAS, SDK_UAS_AWS,
                         SDK_UAS_GCP, BENIGN_GEO, CORP_DOMAIN, GCP_PROJECT, AWS_ACCOUNTS)
import emitters as em

WORK_START, WORK_END = 8, 19

AWS_USER = ["ConsoleLogin","GetUser","ListUsers","ListRoles","ListPolicies",
            "ListAttachedUserPolicies","AssumeRole"]
AWS_ADMIN = AWS_USER + ["CreateUser","AttachUserPolicy","CreateRole","AttachRolePolicy",
                        "AddUserToGroup","CreateAccessKey","ChangePassword","TagRole","CreatePolicy"]
AWS_SVC = ["AssumeRole","GetSessionToken","ListRoles","GetUser"]
AZ_USER = ["Sign-in activity","Update user","Add member to group"]
AZ_ADMIN = AZ_USER + ["Add user","Add member to role","Create group",
                      "Update conditional access policy","Reset user password"]
GCP_USER = ["google.login.LoginService.loginSuccess","GetIamPolicy",
            "google.iam.admin.v1.ListRoles","google.iam.admin.v1.ListServiceAccounts"]
GCP_ADMIN = GCP_USER + ["SetIamPolicy","google.iam.admin.v1.CreateServiceAccount",
                        "google.iam.admin.v1.CreateRole","google.iam.admin.v1.UpdateRole"]


def _ts(day0, day, hour, minute, rng):
    return (day0 + dt.timedelta(days=day, hours=hour, minutes=minute,
            seconds=rng.randint(0,59))).strftime("%Y-%m-%dT%H:%M:%SZ")

def _labels(rng, mild=False):
    if mild and rng.random() < 0.5:
        return ml_label(True, "Normal", round(rng.uniform(0.30, 0.45), 2))
    return ml_label(False, "Normal", round(rng.uniform(0.02, 0.15), 2))

def _pick_cloud(env, user, role, rng):
    if role == "service_account":
        return rng.choices(["aws","gcp","azure"], weights=[0.5,0.4,0.1])[0]
    if role == "human_admin":
        return rng.choices(["aws","azure","gcp"], weights=[0.5,0.3,0.2])[0]
    home = env.home_cloud[user]
    others = [c for c in ["aws","azure","gcp"] if c != home]
    return rng.choices([home]+others, weights=[0.85,0.1,0.05])[0]

def gen_user_day(env, day0, day, user, role, aws, az, gcp):
    rng = env.rng
    upn = env.upn(user); display = user.replace(".", " ").title()
    account = AWS_ACCOUNTS[0]; is_service = role == "service_account"
    n = rng.randint(3,9) if role == "human_admin" else (rng.randint(2,6) if is_service else rng.randint(1,5))
    for _ in range(n):
        hour = rng.randint(0,23) if is_service else (
            rng.randint(WORK_START, WORK_END-1) if rng.random()<0.9 else rng.choice([7,19,20]))
        ts = _ts(day0, day, hour, rng.randint(0,59), rng)
        mild = (hour < WORK_START or hour >= WORK_END) and not is_service
        cloud = _pick_cloud(env, user, role, rng)
        ip = rng.choice(env.internal_ips) if is_service else env.pick_benign_ip()
        if cloud == "aws":
            menu = AWS_SVC if is_service else (AWS_ADMIN if role=="human_admin" else AWS_USER)
            action = rng.choice(menu)
            ua = rng.choice(SDK_UAS_AWS) if is_service else rng.choice(BROWSER_UAS)
            success = not (action=="ConsoleLogin" and rng.random()<0.05)
            aws.append(em.aws_event(ts=ts, event_name=action, user_name=user,
                account_id=account, ip=ip, ua=ua, success=success, mfa=not is_service,
                principal_type="AssumedRole" if is_service else "IAMUser",
                error_code=None if success else "FailedAuthentication",
                read_only=action.startswith(("List","Get")), labels=_labels(rng, mild or not success)))
        elif cloud == "azure":
            op = rng.choice(AZ_ADMIN if role=="human_admin" else AZ_USER)
            az.append(em.azure_event(ts=ts, operation=op, upn=upn, display_name=display,
                ip=ip, ua=rng.choice(BROWSER_UAS), success=True, mfa=True,
                country=rng.choice(BENIGN_GEO), compliant=True, managed=True, labels=_labels(rng, mild)))
        else:
            method = rng.choice(GCP_ADMIN if role=="human_admin" else GCP_USER)
            pe = env.gcp_principal(user, is_service)
            ua = rng.choice(SDK_UAS_GCP) if is_service else rng.choice(BROWSER_UAS)
            gcp.append(em.gcp_event(ts=ts, method=method, principal_email=pe, ip=ip, ua=ua,
                success=True, severity="NOTICE", labels=_labels(rng, mild)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--split", choices=["train","holdout","all"], default="train",
                    help="which identity split to emit benign activity for")
    ap.add_argument("--out", default="./out")
    args = ap.parse_args()
    env = Environment(seed=args.seed, n_users=args.n_users)
    em.reset_ids(); em.set_run_tag(args.split); os.makedirs(args.out, exist_ok=True)
    day0 = dt.datetime(2026, 6, 1)

    aws, az, gcp = [], [], []
    # Select identity split. Held-out users get benign activity too, but ONLY
    # in the holdout split (used for the test set) — never in training benign.
    # This keeps them from being attack-exclusive (no name/label leak) while
    # still being unseen by the trained model.
    if args.split == "train":
        user_subset = env.train_users
    elif args.split == "holdout":
        user_subset = list(env.holdout_users)
    else:
        user_subset = env.human_names
    roster = ([(u, env.users[u]) for u in user_subset] +
              ([(s, "service_account") for s in env.service_names] if args.split != "holdout" else []))
    for day in range(args.days):
        weekday = (day0 + dt.timedelta(days=day)).weekday()
        for user, role in roster:
            if weekday >= 5 and role != "service_account" and env.rng.random() < 0.85:
                continue
            gen_user_day(env, day0, day, user, role, aws, az, gcp)
    for s in (aws, az, gcp):
        s.sort(key=lambda r: r.get("eventTime") or r.get("time") or r.get("timestamp"))
    sfx = "" if args.split=="train" else f"_{args.split}"
    json.dump(aws, open(f"{args.out}/aws_benign{sfx}.json","w"), indent=2)
    json.dump(az,  open(f"{args.out}/azure_benign{sfx}.json","w"), indent=2)
    json.dump(gcp, open(f"{args.out}/gcp_benign{sfx}.json","w"), indent=2)
    total = len(aws)+len(az)+len(gcp)
    anom = sum(1 for st in (aws,az,gcp) for r in st if r["ml_labels"]["anomaly_flag"])
    print(f"[benign] users={len(env.train_users)} (+{len(env.holdout_users)} held out) "
          f"AWS={len(aws)} Azure={len(az)} GCP={len(gcp)} total={total} "
          f"benign-anomalies={anom} ({100*anom/total:.1f}%) -> {args.out}")

if __name__ == "__main__":
    main()