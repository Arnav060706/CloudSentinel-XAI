"""
build_dataset.py — one command to produce the full labelled corpus.

Shared population (env-seed=42) across all splits so identities/holdouts are
consistent and leakage-free. Produces:
  - training benign  (train users only)
  - test benign      (held-out users — unseen by the model, but benign)
  - attacks fast + slow (victims drawn from the shared population)

Unlike earlier versions of this script, all intermediate per-source datasets
are kept permanently on disk (under dataset/raw/...) instead of being
generated into tmp_* folders and deleted at the end. This makes it possible
to inspect or reuse the train/holdout/attack sources directly for downstream
ML work, without changing the generation logic, seeds, or the final merged
dataset in any way.
"""
import subprocess
import sys
import json
import csv
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed parameters (unchanged from the original implementation)
# ---------------------------------------------------------------------------
ENV_SEED = 42
ATTACK_FAST_SEED = 7
ATTACK_SLOW_SEED = 13
DAYS = 21
NOISE_RATIO = 0.4
REPEATS = 8

CLOUDS = ["aws", "azure", "gcp"]

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT = Path("dataset")
RAW = ROOT / "raw"

DIR_TRAIN = RAW / "train"
DIR_HOLDOUT = RAW / "holdout"
DIR_ATTACKS_FAST = RAW / "attacks_fast"
DIR_ATTACKS_SLOW = RAW / "attacks_slow"

DIR_MERGED = ROOT / "merged"
DIR_LABELS = ROOT / "labels"
DIR_METADATA = ROOT / "metadata"

ALL_DIRS = [
    DIR_TRAIN, DIR_HOLDOUT, DIR_ATTACKS_FAST, DIR_ATTACKS_SLOW,
    DIR_MERGED, DIR_LABELS, DIR_METADATA,
]

# Each raw source directory, paired with the filename suffix that the
# corresponding generator script produces (this mapping is unchanged from
# the original implementation — only the directory each source lives in has
# changed).
SOURCE_SUFFIXES = [
    (DIR_TRAIN, "benign"),
    (DIR_HOLDOUT, "benign_holdout"),
    (DIR_ATTACKS_FAST, "attack"),
    (DIR_ATTACKS_SLOW, "attack"),
]


def run(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def reset_output_tree():
    """Clear out only the folders this script itself generates
    (raw/, merged/, labels/, metadata/). We deliberately do NOT
    rmtree the whole dataset/ root — that would also wipe out anything
    else a user has placed under dataset/ (e.g. real IAM logs kept at
    dataset/real/ for validate_realism.py), which is not this script's
    data to delete."""
    for d in ALL_DIRS:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def generate_raw_splits():
    """Run the (unmodified) generator scripts directly into the permanent
    raw/ subfolders instead of tmp_* folders."""
    run([sys.executable, "generate_benign.py",
         "--days", str(DAYS), "--seed", str(ENV_SEED),
         "--split", "train", "--out", str(DIR_TRAIN)])

    run([sys.executable, "generate_benign.py",
         "--days", str(DAYS), "--seed", str(ENV_SEED),
         "--split", "holdout", "--out", str(DIR_HOLDOUT)])

    run([sys.executable, "generate_attacks.py",
         "--seed", str(ATTACK_FAST_SEED), "--env-seed", str(ENV_SEED),
         "--pace", "fast", "--noise-ratio", str(NOISE_RATIO),
         "--repeats", str(REPEATS), "--days", str(DAYS), "--out", str(DIR_ATTACKS_FAST)])

    run([sys.executable, "generate_attacks.py",
         "--seed", str(ATTACK_SLOW_SEED), "--env-seed", str(ENV_SEED),
         "--pace", "slow", "--noise-ratio", str(NOISE_RATIO),
         "--repeats", str(REPEATS), "--days", str(DAYS), "--out", str(DIR_ATTACKS_SLOW)])


def event_sort_key(record):
    return record.get("eventTime") or record.get("time") or record.get("timestamp")


def merge_cloud_logs(cloud):
    """Merge one cloud's events across all raw sources, in the same order
    as the original implementation."""
    merged = []
    for src_dir, suffix in SOURCE_SUFFIXES:
        fn = src_dir / f"{cloud}_{suffix}.json"
        if fn.exists():
            with open(fn) as f:
                merged.extend(json.load(f))
    merged.sort(key=event_sort_key)
    return merged


def write_merged_logs():
    """Write dataset/merged/<cloud>_logs.json — identical in content and
    ordering to the original dataset/<cloud>_logs.json output."""
    for cloud in CLOUDS:
        merged = merge_cloud_logs(cloud)

        with open(DIR_MERGED / f"{cloud}_logs.json", "w") as f:
            json.dump(merged, f, indent=2)


def collect_labels():
    rows = []
    for src_dir in (DIR_ATTACKS_FAST, DIR_ATTACKS_SLOW):
        fn = src_dir / "attack_labels.csv"
        if fn.exists():
            with open(fn, newline="") as f:
                rows += list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("No attack labels were generated.")

    with open(DIR_LABELS / "ground_truth_labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def count_total_events():
    total = 0
    for cloud in CLOUDS:
        with open(DIR_MERGED / f"{cloud}_logs.json") as f:
            total += len(json.load(f))
    return total


def write_stats(rows):
    total_events = count_total_events()
    malicious = sum(1 for r in rows if r["anomaly_flag"] == "True")
    techniques = sorted(set(r["technique_id"] for r in rows if r["technique_id"]))
    holdout_attacks = sum(
        1 for r in rows
        if r.get("victim_split") == "holdout" and r["anomaly_flag"] == "True"
    )

    with open(DIR_METADATA / "DATASET_STATS.txt", "w") as f:
        f.write("CloudSentinel-XAI labelled multi-cloud IAM dataset (leakage-safe)\n")
        f.write(f" total events={total_events}\n"
                f" malicious (labelled)={malicious} ({100*malicious/total_events:.1f}% base rate)\n")
        f.write(f" MITRE techniques={len(techniques)}: {', '.join(techniques)}\n")
        f.write(f" attacks on held-out (unseen) victims={holdout_attacks}  (generalization test set)\n")

    return total_events, malicious, techniques, holdout_attacks


def write_build_config():
    """Optional reproducibility record of all seeds/parameters used."""
    config = {
        "env_seed": ENV_SEED,
        "attack_seeds": {"fast": ATTACK_FAST_SEED, "slow": ATTACK_SLOW_SEED},
        "repeats": REPEATS,
        "days": DAYS,
        "noise_ratio": NOISE_RATIO,
    }
    with open(DIR_METADATA / "build_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    reset_output_tree()
    generate_raw_splits()
    write_merged_logs()

    rows = collect_labels()
    total_events, malicious, techniques, holdout_attacks = write_stats(rows)
    write_build_config()

    print(f"\nDONE -> {ROOT}/ | {total_events} events, "
          f"{malicious} malicious ({100*malicious/total_events:.1f}%), "
          f"{len(techniques)} techniques, {holdout_attacks} held-out-victim attack events")


if __name__ == "__main__":
    main()