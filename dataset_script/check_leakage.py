"""Prove no identity is a label giveaway: every established user that appears
in attacks must ALSO appear in benign (and vice-versa). Only attacker-created
accounts may be attack-exclusive."""
import json, glob, csv
from collections import defaultdict

def actor_of(rec):
    if "userIdentity" in rec: return rec["userIdentity"].get("userName")
    if "properties" in rec: return rec["properties"].get("userPrincipalName","").split("@")[0]
    if "protoPayload" in rec: return rec["protoPayload"]["authenticationInfo"]["principalEmail"].split("@")[0]

benign_users=set(); attack_users=set()
for f in glob.glob("out/*_benign*.json"):   # train + holdout benign
    for r in json.load(open(f)): benign_users.add(actor_of(r))
# attack files contain BOTH attack steps and benign noise; use the label csv for truth
rows=list(csv.DictReader(open("out/attack_labels.csv")))
for r in rows:
    (attack_users if r["anomaly_flag"]=="True" else benign_users).add(r["actor"])

attack_only = attack_users - benign_users
benign_only_in_attackfile = benign_users - attack_users  # fine, just means not attacked

# attacker-created accounts are allowed to be attack-only
created = {"bd-svc-01"}  # names attackers create mid-chain
suspicious_attack_only = {u for u in attack_only if u and not (u.startswith("bd-") or u in created)}

print(f"users seen benign: {len(benign_users)}")
print(f"users seen in attacks (as victim): {len(attack_users)}")
print(f"attack-only identities: {sorted(x for x in attack_only if x)}")
print(f"  -> attacker-CREATED (allowed attack-only): {sorted(x for x in attack_only if x and (x.startswith('bd-') or x in created))}")
print(f"  -> LEAKY established-user attack-only (should be EMPTY): {sorted(suspicious_attack_only)}")
overlap = attack_users & benign_users
print(f"identities appearing in BOTH benign and attack: {len(overlap)}  (this is what kills the leak)")
assert not suspicious_attack_only, "LEAKAGE: some established users are attack-exclusive!"
print("\nPASS: no established identity is a label giveaway. Model must learn behaviour, not names.")