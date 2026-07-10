"""
build_dataset.py — one command to produce the full labelled corpus.

Shared population (env-seed=42) across all splits so identities/holdouts are
consistent and leakage-free. Produces:
  - training benign  (train users only)
  - test benign      (held-out users — unseen by the model, but benign)
  - attacks fast + slow (victims drawn from the shared population)
"""
import subprocess, sys, os, json, glob, csv, shutil
OUT="dataset"; ENV_SEED=42
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
def run(c): print("  $"," ".join(c)); subprocess.run(c, check=True)

run([sys.executable,"generate_benign.py","--days","14","--seed",str(ENV_SEED),"--split","train","--out","tmp_bt"])
run([sys.executable,"generate_benign.py","--days","14","--seed",str(ENV_SEED),"--split","holdout","--out","tmp_bh"])
run([sys.executable,"generate_attacks.py","--seed","7","--env-seed",str(ENV_SEED),"--pace","fast","--noise-ratio","0.3","--repeats","5","--out","tmp_af"])
run([sys.executable,"generate_attacks.py","--seed","13","--env-seed",str(ENV_SEED),"--pace","slow","--noise-ratio","0.3","--repeats","5","--out","tmp_as"])

for cloud in ["aws","azure","gcp"]:
    merged=[]
    for d,suf in [("tmp_bt","benign"),("tmp_bh","benign_holdout"),("tmp_af","attack"),("tmp_as","attack")]:
        fn=f"{d}/{cloud}_{suf}.json"
        if os.path.exists(fn): merged+=json.load(open(fn))
    merged.sort(key=lambda r:r.get("eventTime") or r.get("time") or r.get("timestamp"))
    json.dump(merged, open(f"{OUT}/{cloud}_logs.json","w"), indent=2)

rows=[]
for d in ["tmp_af","tmp_as"]:
    fn=f"{d}/attack_labels.csv"
    if os.path.exists(fn): rows+=list(csv.DictReader(open(fn)))
with open(f"{OUT}/ground_truth_labels.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

tot=sum(len(json.load(open(f"{OUT}/{c}_logs.json"))) for c in ["aws","azure","gcp"])
mal=sum(1 for r in rows if r["anomaly_flag"]=="True")
techs=sorted(set(r["technique_id"] for r in rows if r["technique_id"]))
hold=sum(1 for r in rows if r.get("victim_split")=="holdout" and r["anomaly_flag"]=="True")
with open(f"{OUT}/DATASET_STATS.txt","w") as f:
    f.write("CloudSentinel-XAI labelled multi-cloud IAM dataset (leakage-safe)\n")
    f.write(f" total events={tot}\n malicious (labelled)={mal} ({100*mal/tot:.1f}% base rate)\n")
    f.write(f" MITRE techniques={len(techs)}: {', '.join(techs)}\n")
    f.write(f" attacks on held-out (unseen) victims={hold}  (generalization test set)\n")
for d in ["tmp_bt","tmp_bh","tmp_af","tmp_as"]: shutil.rmtree(d)
print(f"\nDONE -> {OUT}/ | {tot} events, {mal} malicious ({100*mal/tot:.1f}%), "
      f"{len(techs)} techniques, {hold} held-out-victim attack events")