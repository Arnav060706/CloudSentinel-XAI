# CloudSentinel-XAI — Synthetic Multi-Cloud IAM Dataset Generators

Two research-grade generators that synthesize a large, **leakage-safe**,
MITRE-ATT&CK-annotated corpus of AWS / Azure / GCP IAM audit logs, calibrated
to the company's real logs. Train and evaluate CloudSentinel-XAI without live
cloud accounts.

## Headline properties (why this is defensible for a paper)
- **Scale:** a procedurally-generated organization (default 300 users, 25
  services, ~630 IPs incl. 40 Tor + hosting + foreign) over 14 days →
  ~12k events. Configurable via `--n-users`, `--days`.
- **Calibrated to real data:** every user/IP/UA/region/account/category shape
  is drawn from the company's real logs; the real identities and IPs are
  always included, then the population is expanded around them.
  `validate_realism.py` shows the overlap (23/24 real AWS event types, 16/18
  real IPs, all real threat categories reproduced).
- **NO LABEL LEAKAGE (the important fix):** no established identity is a label
  giveaway. Attack scenarios compromise **randomly-drawn legitimate users**,
  so across the corpus every user appears both benign and (sometimes)
  attacked. Only accounts an *attacker creates mid-chain* are attack-exclusive
  — which is realistic and is itself signal. `check_leakage.py` asserts this
  (currently: 0 leaky identities).
- **Generalization test built in:** ~15% of identities are **held out** of
  training benign data and only appear in the test split (benign + attacked),
  so you can report detection on principals the model never trained on.
- **MITRE ATT&CK annotated:** 17 techniques across 9 tactics, going beyond the
  company sample (adds Defense Evasion, Lateral Movement, Exfiltration, Impact).
- **Schema-correct:** parses cleanly through your real `ParserPipeline`
  (validated: 12181/12181, 0 failures).
- **Realistic base rate:** ~2–8% malicious (real environments are mostly
  benign) — not artificially balanced.

## Files
| File | Purpose |
|------|---------|
| `env_profile.py` | `Environment` — procedural, seeded, label-neutral population |
| `emitters.py` | Emit records in exact AWS/Azure/GCP schemas |
| `generate_benign.py` | Benign baseline; `--split train\|holdout\|all` |
| `attack_scenarios.py` | 4 MITRE cross-cloud kill chains (edit/extend here) |
| `generate_attacks.py` | Renders attacks on random legit victims + benign noise |
| `check_leakage.py` | **Asserts no identity is a label giveaway** |
| `validate_realism.py` | Synthetic-vs-real statistical fidelity table |
| `build_dataset.py` | One command → full labelled corpus in `dataset/` |

## Quick start
```bash
python build_dataset.py       # -> dataset/  (~12k events, ~2.7% malicious, 17 techniques)
python check_leakage.py       # prove no username leaks the label
python validate_realism.py    # fidelity vs the real company logs
```

## The two research knobs that matter
- **Population size:** `--n-users`, `--days` scale the org (match real-org
  scale; don't inflate to obviously-synthetic millions).
- **Pacing:** `generate_attacks.py --pace {fast,slow,mixed}`. `slow` spaces
  attack steps 10 min–1.5 h apart — the **low-and-slow APT** your cross-cloud
  multiplier is designed to defeat. `build_dataset.py` generates both a fast
  and a slow pass so you can show detection holds regardless of pacing.

## Anti-leakage design (how it works)
1. Identities are role-typed but label-neutral (no "villain" names).
2. Each attack scenario declares a `victim_role`; the generator draws a
   **random legitimate user** of that role as the victim, using the **same
   seeded `Environment`** (`--env-seed`) the benign data used — so victims are
   real members of the population.
3. Held-out users get benign activity in the *holdout* split (test), never in
   training — unseen by the model but still benign, so they're not
   attack-exclusive.
4. `user_id`/principal must be dropped from the model's features X (behaviour
   only) — enforced in the fixed `feature_extractor.py`.

## Ground-truth label columns
`event_id, scenario, actor, cloud, action, timestamp, anomaly_flag,
threat_category, severity_score, tactic, technique_id, technique_name, infra,
note, victim_split, created`  (`victim_split` = train|holdout for the
generalization split; `created` = identity name this step creates, e.g.
`bd-svc-01`, empty for steps that don't create an identity — also written into
the event's `requestParameters`/`request` so it's real log content, not just a
label, and read back by `check_leakage.py` to verify it's correctly
attack-exclusive)

## Reproducibility
`--seed` (attack randomness) and `--env-seed` (population identity) are both
recorded; same seeds → identical dataset. Record them in the paper.

---

## Next Steps

**1. Train the models & flip out of bypass mode.**
Read `dataset/`, engineer features with the fixed `feature_extractor.py`
(drop `user_id`, `timestamp`, `raw_log` from X — behaviour only), train the
Isolation Forest (on benign only) and the XGBoost ATT&CK-phase classifier (on
labelled data), pickle both into `models/`. The pipeline auto-upgrades from
bypass to real scoring. *Next artifact to build.*

**2. Run the evaluation the paper needs.**
- Baselines: plain Isolation Forest, plain XGBoost, a rule-based detector, and
  a GNN baseline (to compare against the closest prior work honestly).
- **Ablations** (this proves your contributions): turn off identity stitching,
  the cross-cloud multiplier, the baseline-relative correction, and the
  faithfulness gate — measure the drop from each.
- Metrics at the realistic base rate: PR-AUC, **false-positives per
  analyst-hour**, detection latency per kill-chain stage, identity-stitching
  precision/recall.
- **Generalization experiment:** report detection on held-out (unseen)
  victims vs seen users — proof the model learned behaviour, not identities.
- **Pacing experiment:** detection on fast vs slow attacks — proof the
  cross-cloud multiplier defeats low-and-slow evasion.

**3. Adversarial evaluation.**
Attack your own system: UA rotation to break stitching, calibration
poisoning, prompt injection against the LLM narrative, and slow-pacing that
outlasts the 60s window (motivates the persistent-risk term).

**4. Persistent trust & risk (from the review recommendations).**
Add the per-entity, cloud-independent, slowly-recovering trust score as a real
SQLite column; trigger XAI on sustained low trust and explain *why trust
dropped* over time (trust-trajectory attribution) — a novel explanation type.

**5. Realism validation for the paper.**
Expand `validate_realism.py` into a formal fidelity table (event-type
distributions, inter-arrival times, error rates, principals/session) with a
statistical distance (e.g. KS / Jensen-Shannon) vs the real logs. This is the
table that disarms "it's just synthetic" at review.

**6. Release as an artifact.**
No strong public labelled multi-cloud IAM-APT dataset exists — package the
generators + a frozen dataset + seeds for the ACSAC/RAID artifact track. This
is itself a contribution and makes the work citable by the next team.

**7. Scale/perf hardening (if reviewers push).**
Attribute-blocking / LSH for the Tier-2 fuzzy stitch (turns the O(E) scan
near-linear); benchmark per-event latency vs population size.