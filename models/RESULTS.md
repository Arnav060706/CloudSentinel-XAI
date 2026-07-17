# RESULTS.md — paper-facing results reference

**Purpose.** A single place to look up every number that might go in the paper, with
the caveats attached to it, so the write-up doesn't have to reconstruct context from
`models/README.md`.

**Relationship to `models/README.md`.** `README.md` remains the authoritative
*engineering log* — what was tried, what broke, why. This file is the *paper-facing
results reference*: the same numbers, organised by claim, with the reporting rules
that must travel with them. If the two ever disagree, `README.md` + the CSVs win;
fix this file.

**Status: PRELIMINARY. Nothing here is final-paper-ready yet.** Single seed (n=1), no
confidence intervals, configuration not yet frozen. See §8.

---

## 0. Reading rules — read before quoting any number from this file

These are the five ways someone (including you, in three months) will misread this
file. Each is a real trap that already caught us at least once.

1. **Never quote a PR-AUC without its base rate.** PR-AUC is bounded below by the
   positive base rate. Ours is ~9.7%, so the no-skill floor is ~0.097 — not 0. A
   PR-AUC of 0.237 is 2.45× floor, not "0.237 out of 1.0". See §0.2.
2. **PR-AUC is not the precision you'd ship with.** It averages precision across
   *every* threshold, including ones you'd never use. At its chosen operating point
   C2 gives precision 0.69, not 0.24. See §0.3.
3. **Campaign recall is gameable and must never appear without precision beside
   it.** A campaign counts as caught if *any one* of its ~17 steps is flagged, so
   flooding trivially maximises it. Row B1 hits campaign recall **1.000 (32/32) at
   precision 0.116** — i.e. by flagging ~three quarters of all traffic. See §3.
4. **Rows compared at their own max-F1 thresholds are NOT comparable.** This misled
   us in both directions (§5). Cross-row claims must come from the matched-precision
   table (§4).
5. **[cal] and [frozen] numbers are from DIFFERENT DATASETS — never mix them in one
   sentence.** C2's event recall is **0.144 on frozen** (§2) and **0.205 on cal**
   (§3). Both are correct. Quoting one as the other is wrong.

### 0.1 What "positive" means in this project

- **Unit of evaluation: one normalized IAM log event** (one API call — `ConsoleLogin`,
  `AssumeRole`, `CreateAccessKey`, …).
- **Positive (y=1): the event is a step in a scripted attack kill chain.** Set
  mechanically in `models/scoring_utils.py`:
  `y_true.append(int(category not in ("Normal", "", None)))` where `category` is
  `ml_labels.threat_category` from the raw log.
- **Negative (y=0): a benign event** — *including* deliberately odd-but-legitimate
  activity. `generate_benign.py` marks a fraction of off-hours activity with
  `anomaly_flag=True` but keeps `threat_category="Normal"`; those count as
  **negatives**. The model gets no credit for flagging them and they count as false
  positives. This is the strict choice and makes our numbers look *worse* than a
  looser labelling would — **say so in the paper**, it is a point in our favour.
- **What is actually ranked:** for row A, the per-event anomaly score. For rows
  B/C/D it is `scaled_score` — the RiskEngine's score for the **entity** the event
  belongs to at that moment (decayed sum of recent anomaly scores × cross-cloud
  multiplier). So the question each row answers is *"at the instant this event
  occurred, did the system consider the actor behind it risky?"* Label is per-event;
  score is per-actor-at-that-time.

### 0.2 Base rates and no-skill floors

| Set | Events | Attack events | Base rate | **No-skill PR-AUC floor** |
|---|---|---|---|---|
| **Frozen** (`holdout` + `attacks_slow`) | 2,732 | 264 | **9.66%** | **0.0966** |
| **Cal** (`holdout_cal` + `attacks_cal`) | 2,660 | 264 | **9.92%** | **0.0992** |

Use **0.0966** for every [frozen] figure and **0.0992** for every [cal] figure.

**Maximum achievable lift = 1/base rate.** At 9.66% a *perfect* detector maxes out at
**10.35× floor**. So C2's 2.45× uses ~24% of the available headroom — state it that
way rather than implying 2.45× is near a ceiling.

**Do not "improve" results by raising the base rate.** Precision transforms as
`precision = TPR·π / (TPR·π + FPR·(1−π))`. Holding the detector fixed and moving π
from 9.66% → 30% would lift C2's precision 0.69 → ~0.90 and B2's 0.20 → ~0.50 with
*zero* change to the model. ROC-AUC would not move at all. Reviewers know this trick.
Real cloud IAM base rates are far *lower* (≪1%), where C2's precision would fall to
~0.17 (π=1%) or ~0.02 (π=0.1%). Cite **Axelsson, "The Base-Rate Fallacy and its
Implications for the Difficulty of Intrusion Detection," ACM CCS 1999 / TISSEC 2000**
and frame our figures as an upper bound on real-world precision.

### 0.3 ROC-AUC vs PR-AUC — why ours disagree

- **ROC-AUC** = P(a random attack event is ranked above a random benign event).
  Base-rate invariant. C2 = 0.709 → 71% of the time.
- **PR-AUC** = precision averaged across all recall levels. Base-rate dependent.

There are ~9.3 benign events per attack event, so even a modest false-positive *rate*
floods the alert queue relative to just 264 positives. ROC-AUC hides this (its axis
is normalised against the large benign pool); PR-AUC exposes it. **PR-AUC is the
honest headline metric for this problem** — that is why `README.md` says to trust it.

---

## 1. Dataset splits and what each may inform

| Split | Role |
|---|---|
| `Datasets/Train_iso` | Isolation Forest training (benign only) |
| `Datasets/attacks_fast` | XGBoost **training** — leaks into any XGB decision |
| `Datasets/holdout_cal` + `Datasets/attacks_cal` | **[cal]** all tuning/selection/exploration |
| `Datasets/holdout` + `Datasets/attacks_slow` | **[frozen]** final test — consume **once**, post-freeze |

**Ordering discipline (non-negotiable):** all selection happens on [cal]; then the
config is frozen and committed; then **one** pass over [frozen] produces every
reportable table. Anything discovered afterwards goes in Limitations, not the table.
`models/evaluate_campaign_recall.py` enforces this in code — it refuses
`--attack-dir Datasets/attacks_slow` without an explicit `--frozen-run`.

---

## 2. Table 1 — Main ablation **[frozen]** *(preliminary)*

**Claim it supports:** the graph layer is what works; the oracle gap localizes the
remaining error.

Source: `models/README.md` "Phase 2: label-swept, frozen threshold".
Set: `holdout` + `attacks_slow`. Floor = **0.0966**. Threshold swept on cal, frozen.

| Row | Config | ROC-AUC | PR-AUC | **×floor** | Prec | Recall | TP / FP |
|---|---|---|---|---|---|---|---|
| **A †** | IF alone, per-event | 0.523 | 0.093 | 0.96× | 0.113 | 0.867 | — |
| B1 | IF + risk, no graph | 0.531 | 0.095 | 0.98× | 0.109 | 0.921 | 243 / 1986 |
| C1 | IF + risk + graph | 0.630 | 0.198 | 2.05× | 0.606 | 0.227 | 60 / 39 |
| D1 | IF + oracle identity *(diag.)* | 0.773 | 0.480 | 4.97× | 0.747 | 0.549 | 145 / 49 |
| **B2** | **XGB + risk, no graph** | 0.678 | 0.144 | **1.49×** | 0.198 | 0.545 | 144 / 583 |
| **C2** | **XGB + risk + graph** | **0.709** | **0.237** | **2.45×** | **0.691** | 0.144 | 38 / 17 |
| **D2** | **+ oracle identity** *(diag.)* | 0.774 | 0.411 | **4.25×** | 0.793 | 0.333 | 88 / 23 |

> **† ROW A IS A DATASET MISMATCH — FIX BEFORE PUBLISHING.** Row A's published
> figures (ROC 0.523 / PR-AUC 0.093, and precision 0.113 / recall 0.867) come from
> `README.md`'s "Current results" section, which was measured on `holdout` +
> **`attacks_fast`** — *not* the frozen `attacks_slow` set every other row uses. As
> printed, **Table 1 mixes two datasets**, violating rule 4 in §0.
>
> The fix is easy and costs no new frozen run: `README.md` records that **row B is "an
> unchanged monotonic rescale of row A for single-event entities"**, and a monotonic
> rescale preserves ROC-AUC and PR-AUC *exactly*. So **on the frozen set, row A's
> AUCs are identical to B1's: ROC 0.531 / PR-AUC 0.095 (0.98× floor)**. Either quote
> those, or drop row A and let B1 serve as the per-event baseline. (Precision/recall
> differ, since those are threshold-dependent and the rescale moves the threshold.)
>
> This is worth a moment's thought rather than a mechanical patch: it is exactly the
> class of error §0 rule 4 exists to catch, and it survived into a table that was
> about to go in a draft.

**Rows D1/D2 are diagnostics, not system results.** They substitute ground-truth
cross-cloud identity for learned stitching. They are an upper bound and must be
labelled as such wherever they appear.

**The demo-draft subset** the paper's Table 1 uses is A, B2, C2, D2 (+ R and E1–E3
still missing — see §8).

### 2.1 Drop-in LaTeX

```latex
\begin{table}[t]
\centering
\caption{Main ablation on the held-out slow-pace attack set
(\texttt{attacks\_slow} + benign holdout; 264 attack events, no-skill
PR-AUC $=0.097$). Stages are added cumulatively. \textbf{$\times$floor}
is PR-AUC lift over the no-skill baseline. Row~D2 uses ground-truth
cross-cloud identity and is a diagnostic \emph{upper bound}, not a
deployable configuration. \emph{Preliminary: single seed, no confidence
intervals, configuration not yet frozen.}}
\label{tab:main-ablation}
\begin{tabular}{@{}llccc@{}}
\toprule
Row & Configuration & ROC-AUC & PR-AUC & $\times$floor \\
\midrule
A  & IF alone, per-event            & 0.531 & 0.095 & 0.98$\times$ \\
B2 & XGB + risk, no graph           & 0.678 & 0.144 & 1.49$\times$ \\
C2 & XGB + risk + graph             & \textbf{0.709} & \textbf{0.237} & \textbf{2.45$\times$} \\
D2 & \quad + oracle identity \emph{(diag.)} & 0.774 & 0.411 & 4.25$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

*(Row A above uses the frozen-set values 0.531 / 0.095 / 0.98×, per the † note — NOT
the `attacks_fast` figures 0.523 / 0.093.)*

### 2.2 Drop-in results paragraph

> Table~\ref{tab:main-ablation} reports the main ablation, adding each pipeline stage
> cumulatively on the held-out slow-pace attack set. On a per-event basis the
> unsupervised Isolation Forest sits essentially at the no-skill floor (PR-AUC 0.095,
> 0.98×), confirming that individual IAM events carry little standalone anomaly
> signal. Replacing it with the supervised XGBoost attack-type signal and aggregating
> through the decayed risk engine (B2) lifts PR-AUC to 0.144 (1.49× floor). Adding the
> cross-cloud identity-stitching graph layer (C2) is the decisive step: PR-AUC rises
> to 0.237 (2.45× floor) and ROC-AUC to 0.709 — a 65% relative PR-AUC gain over the
> no-graph configuration — indicating that the graph layer, not the base classifier,
> is what separates attacks in this regime. The oracle-identity row (D2; 0.411 PR-AUC,
> 4.25× floor) substitutes ground-truth cross-cloud identity for the learned stitching
> and serves as a diagnostic upper bound: the gap between C2 and D2 localizes
> essentially all remaining error to imperfect identity resolution rather than to the
> risk-scoring concept itself. Because the positive base rate is only 9.7%, absolute
> PR-AUC should be read against the no-skill floor (0.097) — the ×floor column —
> rather than against benchmarks reported on balanced or high-base-rate corpora.

### 2.3 Caveat block

> **Preliminary results.** These figures reflect a single dataset seed (n=1);
> multi-seed mean ± std and bootstrap confidence intervals on PR-AUC are deferred to
> the camera-ready. The pipeline configuration is not yet frozen, so point values may
> shift. A rule-based (non-ML) baseline and IF/XGBoost fusion variants are in progress
> and not yet included.

> **Synthetic-data scope.** The evaluation set is synthetic, generated from four MITRE
> ATT&CK scenario templates; results therefore demonstrate recognition of known
> scripted attack patterns, not proven generalization to novel techniques.

---

## 3. Table 2 — Campaign-level (kill-chain) detection **[cal]**

**Why this exists.** Event-level recall asks "what fraction of attack *steps* did we
flag?". Operationally the question is "did we catch the *intrusion*?" — for each
campaign, was **any** step flagged? The 264 attack events are the steps of only **32
campaigns** (8 `--repeats` × 4 scenarios), so event recall can badly understate real
detection capability.

Script: `models/evaluate_campaign_recall.py` → `models/campaign_recall_cal.csv`.
Set: `holdout_cal` + `attacks_cal` (2,660 events, 264 attack, 9.92%).

### 3.1 Campaign identity — the methodology that makes this valid

`generate_attacks.py` runs `for _ in range(repeats): for scen in ALL_SCENARIOS:` and
draws a **fresh random victim per campaign**, but emits only `scenario` and `actor` —
**there is no campaign/repeat id**. The draw sometimes picks the *same* victim twice
for the *same* scenario, so **`(scenario, actor)` is NOT a valid campaign key**:

- `attacks_cal`: 31 distinct pairs vs 32 real campaigns — `alice.chen` has 34 = 2×17
  `cross_cloud_apt` events.
- `attacks_slow`: 29 distinct pairs vs 32 — `yuki.park` 34 = 2×17, `svc-ci-pipeline`
  15 = 3×5.

Grouping by `(scenario, actor)` would **merge** campaigns, and a merged campaign needs
only one detected event to count as caught — **inflating campaign recall**. Campaigns
are therefore also split on a **time gap**, justified by measurement:

- largest observed **intra**-campaign gap: **1.49h** (slow pacing ≤ ~1.5h between steps)
- smallest observed **inter**-campaign gap: **34.1h** (`alice.chen`), ~142h (`svc-ci-pipeline`)
- **`--gap-hours 6`** sits ~4× above the largest intra gap and ~5.7× below the smallest
  inter gap.

**The clustering is validated, not trusted:** the true count is known exactly from the
generator (32) and the script **fails loudly** if clustering doesn't recover it. It
recovered **32/32**.

### 3.2 Results at each row's own max-F1 threshold

⚠️ **These rows are NOT comparable to each other** (each uses its own threshold). Use
§4 for cross-row claims. This table is only for the *within-row* event→campaign
comparison.

| Row | thr | Event recall | Event prec | FP | Campaign recall | ×event | Median detect step |
|---|---|---|---|---|---|---|---|
| B1. IF, no graph | 0.318 | 0.951 | **0.116** | 1916 | **1.000** (32/32) | 1.05× | 1 (20%) |
| C1. IF + graph | 0.425 | 0.284 | 0.383 | 121 | 0.469 (15/32) | 1.65× | 5 (60%) |
| D1. IF + oracle | 0.425 | 0.534 | 0.476 | 155 | 0.969 (31/32) | 1.81× | 3 (60%) |
| B2. XGB, no graph | 0.367 | 0.549 | 0.192 | 609 | 0.969 (31/32) | 1.76× | 2 (20%) |
| **C2. XGB + graph** | 0.517 | 0.205 | 0.562 | 42 | 0.469 (15/32) | **2.29×** | 5 (71%) |
| D2. XGB + oracle | 0.513 | 0.352 | 0.604 | 61 | 0.875 (28/32) | 2.48× | 4 (60%) |

**Finding 1 — the hypothesis is directionally confirmed.** Campaign recall exceeds
event recall in every row (1.05×–2.48×). Event-level recall genuinely understates
detection.

**Finding 2 — but campaign recall is gameable, and B1 proves it.** B1 reaches
**32/32 campaigns at precision 0.116** — 1,916 false positives, ~91/day. A detector
that alerts on ~three quarters of traffic "catches" every campaign. **Campaign recall
must never be reported without precision or an alert budget.**

### 3.3 Per-scenario campaign detection (max-F1 thresholds)

| Scenario | B1 | C1 | D1 | B2 | **C2** | D2 |
|---|---|---|---|---|---|---|
| `cross_cloud_apt_credential_theft` | 8/8 | 7/8 | 7/8 | 8/8 | **7/8** | 7/8 |
| `insider_privilege_abuse` | 8/8 | 2/8 | 8/8 | 8/8 | **2/8** | 7/8 |
| `logging_tamper_and_destruction` | 8/8 | 3/8 | 8/8 | 8/8 | **3/8** | 7/8 |
| `service_account_key_abuse` | 8/8 | 3/8 | 8/8 | 7/8 | **3/8** | 7/8 |

**This is the most paper-valuable table in the file.** C2 catches **7/8 cross-cloud
APT campaigns** and largely misses the single-cloud ones — *exactly what the mechanism
predicts*, since the cross-cloud multiplier can only fire on campaigns that actually
cross clouds. `insider_privilege_abuse` at 2/8 **independently corroborates the LOSO
blind spot** (§6) via a completely different measurement.

Defensible claim: *"the graph layer detects the threat class it was designed for, and
we can show precisely where it does not apply."*

---

## 4. Table 3 — Matched-precision comparison **[cal]** — the honest cross-row result

Each row's operating point is re-derived at a **common event-precision floor**, so
campaign recall is compared at **equal alert quality**. Source:
`models/campaign_recall_matched_cal.csv`.

| Row | @prec ≥ 0.30 | @prec ≥ 0.50 | FP @0.50 | @prec ≥ 0.70 |
|---|---|---|---|---|
| B1. IF, no graph | **unreachable** | **unreachable** | — | **unreachable** |
| B2. XGB, no graph | **unreachable** | **unreachable** | — | **unreachable** |
| C1. IF + graph | 15/32 (0.469) | 2/32 (0.062) | 5 | 1/32 (0.031) |
| **C2. XGB + graph** | 15/32 (0.469) | **15/32 (0.469)** | **52** | 2/32 (0.062) |
| D1. IF + oracle | 31/32 (0.969) | 7/32 (0.219) | 14 | 5/32 (0.156) |
| **D2. XGB + oracle** | 28/32 (0.875) | **28/32 (0.875)** | **69** | 3/32 (0.094) |

**Finding 3 — the no-graph baselines cannot reach even precision 0.30, at any
threshold.** B2's precision *ceiling* is below 0.30. This follows directly from the
benign-outlier problem already documented in `README.md` (some benign events outscore
every attack, so raising the threshold destroys TPs before FPs). **This is a stronger
argument for the graph layer than the PR-AUC gap in Table 1** — the no-graph
configuration is not merely worse, it is *unusable at any acceptable alert quality*.

**Finding 4 — XGBoost's contribution is large, but only visible under fair
comparison.** At precision ≥ 0.50, **C2 catches 15/32 vs C1's 2/32 — 7.5×** more
campaigns at identical alert quality (D2 vs D1: 28/32 vs 7/32, 4×).

**Finding 5 — precision ≥ 0.70 is unreachable for every configuration** (best: D1 at
5/32). The usable operating regime is **precision ≈ 0.5**.

---

## 5. The C1/C2 structural finding (and why it is not a bug)

At their own max-F1 thresholds, C1 and C2 catch **exactly the same 15 campaigns**
(Jaccard **1.000**). Identical results across two different models is a classic bug
signature, so it was checked directly:

| Pair | Jaccard | Verdict |
|---|---|---|
| C1 vs C2 (graph) | **1.000** | IDENTICAL |
| D1 vs D2 (oracle) | 0.903 | DIFFERENT |
| B1 vs B2 (no graph) | 0.969 | DIFFERENT |

**It is not a bug.** If the rows were sharing scores, *all three* pairs would be
identical — they are not. C1/C2 also differ at event level (recall 0.284 vs 0.205,
precision 0.383 vs 0.562, FP 121 vs 42). The plumbing is correct.

**Interpretation — the division of labour, and the paper's cleanest mechanistic
claim:**

- **The graph layer determines *which* campaigns are catchable at all** (the same 15
  light up regardless of the base signal — the cross-cloud multiplier decides).
- **The ML signal determines *at what precision* you can catch them.** XGB holds those
  15 at precision 0.56; IF needs to flood to precision 0.38 for the same 15, and at a
  matched 0.50 floor it collapses to 2.

⚠️ **Methodological warning worth a sentence in the paper:** at their own thresholds,
C1 and C2 looked identical, implying XGBoost added nothing. The matched-precision
comparison shows it adds 7.5×. **The own-threshold comparison was misleading in both
directions.**

---

## 6. Operational framing (best current numbers for a "does this work?" claim)

Over the 21-day cal window, at precision ≈ 0.5:

| Config | Intrusions caught | False positives | **FP / day** |
|---|---|---|---|
| **C2 (deployable)** | **15 / 32** | 52 | **~2.5** |
| D2 (oracle, upper bound) | 28 / 32 | 69 | ~3.3 |
| B2 (no graph, max-F1) | 31 / 32 | 609 | ~29 |
| B1 (no graph, max-F1) | 32 / 32 | 1,916 | ~91 |

*"~2.5 false alerts per day while catching 15 of 32 intrusions"* is far more
compelling than an abstract AUC — and the B1/B2 rows show what the alternative costs.

**Detection latency (honest negative):** C2 first flags a campaign at **median step 5,
71% through the kill chain**, vs B2 at step 2 (20%). The risk engine must accumulate
decayed evidence, so detection lags. Catching an intrusion 71% of the way through may
be operationally too late — **state this**.

---

## 7. Corroboration — leave-one-scenario-out **[Phase 3]**

`models/evaluate_loso.py`, binary `1 − P(Normal)` framing.

| Held-out scenario | ROC-AUC | PR-AUC |
|---|---|---|
| `cross_cloud_apt_credential_theft` | 0.999 | 0.976 |
| `logging_tamper_and_destruction` | 0.973 | 0.560 |
| `service_account_key_abuse` | 0.997 | 0.920 |
| **`insider_privilege_abuse`** | **0.515** | 0.038 |
| **MEAN** | **0.871** | 0.623 |

**State this next to the 0.91 in-distribution accuracy.** 3/4 unseen scenarios
generalize (0.97–0.999); `insider_privilege_abuse` is a genuine blind spot (~random,
47/48 events classified `Normal`) — an insider abusing already-valid privileges is
behaviorally indistinguishable from benign to this feature set. **§3.3's independent
2/8 campaign result corroborates this.** The defensible claim is: strong
generalization to unseen scenarios carrying overt attack signals, and a genuine blind
spot for low-and-slow insider abuse — **not** a blanket "generalizes to novel
techniques".

---

## 8. Limitations / what is NOT done yet

| Item | Status | Note |
|---|---|---|
| **Multi-seed variance** (≥5 seeds, mean ± std) | **missing — highest priority** | n=1 is an anecdote at RAID. Not "5 reruns" — 5 *fresh sealed test sets*. Seeds must be picked **at freeze time**; the "one frozen run" becomes "one run per seed at the locked config". |
| **Bootstrap CIs on PR-AUC** | missing | 264 positives is few; the interval is wide. Resamples existing predictions — cheap, no new data. |
| **Row R — rule-based baseline** | **deferred** | `engines/deterministic_rules.py` is **0 bytes** (as are `identity.py`, `network.py`). Reviewers ask "does ML beat rules?" early. Build/tune on **cal**. Blocked on §8.1: the planned *impossible-travel* rule is not honestly implementable on this data. |
| **Labeled-only features are label proxies** | **known, deferred** | See **§8.1**. Decision taken: fix the generator later; not addressed for the draft. |
| **Rows E1–E3 — IF/XGB fusion** | missing | See plan §4. |
| **Campaign recall on frozen set** | missing | §3/§4 are **cal-set, in-sample thresholds**. Reportable version = single post-freeze pass. |
| **Matched-precision curve** | partial | Only 3 precision floors sampled; a full campaign-PR curve would be stronger. |
| **Config freeze** | not done | Config lives as constructor defaults across 3 files; no freeze artifact yet. |
| **Live-path baseline warmup** | open | Phase 1's `record_baseline()` is offline-eval only; live alerts still use the uncalibrated default threshold. |

### 8.1 KNOWN DATASET ARTIFACT — the "labeled-only" features are label proxies

**Status: measured, acknowledged, deferred.** Decision (2026-07-16): fix the
generator later; not addressed for the draft paper. Recorded here so it is not
rediscovered the hard way — e.g. by a reviewer.

**The measurement** (`holdout_cal` benign n=2291 vs `attacks_cal` attack n=264):

| Field | Benign | Attack |
|---|---|---|
| `geo_country` | **`Unknown` — 2291/2291 (100%)** | real country (DE 82, GB 31, JP 27, IR 18, RU 13, …) — **215/264** |
| `is_known_proxy_or_tor` | **False — 2291/2291 (100%)** | True 44 / False 220 |
| `mfa_authenticated` | True 1531 / False 760 | **False — 264/264 (100%)** |
| `status` | SUCCESS 2286 / FAILED 5 | SUCCESS 248 / FAILED 16 |

(The 105 benign-noise events inside `attacks_cal` are also 100% `Unknown` geo and
100% False proxy — so this is not a benign/attack *file* artifact, it tracks the label.)

**Consequence: a one-line rule `geo_country != "Unknown"` scores ~1.00 precision at
0.814 recall** — beating the entire ML pipeline (C2: precision 0.56, recall 0.20).
That is not a detector; it is the benign emitter never resolving a country while the
attack emitter always does.

**Why this exists.** `feature_extractor.py`'s `include_labeled_only_features=True`
adds `geo_country`, `is_known_proxy_or_tor`, `device_compliant_status`,
`is_internal_ip` to XGBoost's 28-column set, on the README's rationale that they are
"zero-variance on benign-only data but have real, confirmed variance on labeled
attack data." Stated precisely, **variance that exists only on the attack side is the
label**. The rationale is the bug.

**How much does XGBoost actually depend on it?** Less than feared — this is the
mitigating finding:

| Feature | Importance | Rank (of 28) |
|---|---|---|
| `login_result_success` | 0.2805 | 1 |
| `unique_resources_accessed` | 0.2737 | 2 |
| **`mfa_authenticated`** | **0.1020** | **3** |
| **`is_internal_ip`** | 0.0335 | 8 |
| **`device_compliant_status`** | 0.0250 | 9 |
| **`geo_country`** | 0.0073 | **15** |
| **`is_known_proxy_or_tor`** | 0.0050 | 16 |
| | **Σ leaky ≈ 0.173 (~17%)** | |

`geo_country` ranks **15th** — the headline numbers are **not** built on the worst
offender (likely because `login_result_success` / `unique_resources_accessed` already
separate Normal from attack, so the tree never needs it). The real concern is
**`mfa_authenticated` at rank 3**, which is 100% False on attacks — though it is only
a *partial* proxy (33% of benign is also False, so `mfa=False` ⇒ P(attack) ≈ 0.26)
and is arguably realistic: attackers using stolen credentials often cannot satisfy MFA.

**Blocks Row R.** The plan's *impossible-travel* rule requires `geo_country`, but
benign has **no country at all** — so any version of it fires on attacks and never on
benign, i.e. it reads the label. Same for a Tor/proxy rule. An honest Row R can use
only behavioral/temporal signals: `status`, `action`, `timestamp`, `source_cloud`, and
a normalized username (`user_id` is cloud-native — `arn:aws:iam::…:user/mason.khan`
vs `mason.khan@corp-example.com` — so R2 needs SIEM-style username normalization):

- **R1** brute force: ≥N `FAILED` logins → `SUCCESS`, same user, within a window
- **R2** cross-cloud velocity: same username in ≥2 clouds within N minutes
- **R3** privilege grant → immediate use
- **R4** recon burst: ≥N `List*`/`Get*` IAM calls within a window

**When picked back up, the options are:**
1. **Quantify** — retrain XGBoost without the four labeled-only features, re-run the
   cal ablation, and measure what the leak was worth. Cheapest path to certainty.
2. **Drop them** and adopt that as the frozen config.
3. **Fix the generator** (the chosen direction) — have `generate_benign.py` assign
   realistic `geo_country` / proxy / device values so the features carry genuine
   signal. Most correct; requires regenerating datasets, so it must land **well before
   the freeze**.

Note `dataset_script/check_leakage.py` does **not** catch this — it asserts no
*identity* is a label giveaway, not that no *feature* is.

**Config surface that must be frozen** (current values):

| Knob | Value | Location |
|---|---|---|
| `half_life_seconds` | 1800.0 | `risk_engine.py:138` |
| `per_cloud_multiplier` | 3.0 | `risk_engine.py:139` |
| `automation_cloud_multiplier` | 1.7 | `risk_engine.py:140` |
| `recalibration_percentile` | 95.0 | `risk_engine.py:142` |
| `baseline` | 0.05 | `risk_engine.py:137` |
| `window_horizon_seconds` | 60 | `graph_engine.py:190` |
| `tau` (similarity_threshold) | 0.65 | `graph_engine.py:191` |
| `ambiguity_margin` | 0.05 | `graph_engine.py:192` |
| `tier2_lookback_seconds` | 10800 | `graph_engine.py:193` |
| fusion weights | UA 0.45 / proxy 0.25 / type 0.20 / IP 0.10 | `graph_engine.py:146-149` |
| threshold selection | swept max-F1 on cal, frozen | `evaluate_full_pipeline.py:306` |
| feature set | IF 24-col / XGB 28-col | `feature_extractor.py` |
| seeds | env-seed 42; benign 42; attacks fast/slow/cal = 7/13/21 | dataset commands |

---

## 9. Provenance — exact commands

```bash
# [frozen] Table 1 — main ablation (Phase 2 numbers)
python models/evaluate_full_pipeline.py \
    --holdout-dir Datasets/holdout --attack-dir Datasets/attacks_slow \
    --baseline-dir Datasets/holdout_cal \
    --cal-holdout-dir Datasets/holdout_cal --cal-attack-dir Datasets/attacks_cal \
    --cal-baseline-dir Datasets/holdout --out models/ablation_results_baselined.csv

# [cal] Tables 2 + 3 — campaign-level detection + matched precision
python models/evaluate_campaign_recall.py
#   -> models/campaign_recall_cal.csv
#   -> models/campaign_recall_matched_cal.csv

# [Phase 3] LOSO
python models/evaluate_loso.py     # -> models/loso_results.csv
```

**Code state:** `models/evaluate_campaign_recall.py` (new) and a purely additive change
to `evaluate_full_pipeline.py`'s `prepare_set()` (now also returns `event_id`, needed
for the campaign join). Both **uncommitted** at time of writing.

---

## 10. Reporting rules — the do-not list

1. **Do not** quote a PR-AUC without its base rate and ×floor.
2. **Do not** quote campaign recall without precision or FP/day beside it.
3. **Do not** compare rows at their own max-F1 thresholds — use §4.
4. **Do not** mix [cal] and [frozen] numbers in one sentence.
5. **Do not** present D1/D2 (oracle) as system results — they are diagnostic upper bounds.
6. **Do not** raise the base rate to improve PR-AUC.
7. **Do not** re-run [frozen] after any config change — that converts it into a dev set.
8. **Do not** claim "generalizes to novel techniques" — the data is 4 scripted templates,
   and `insider_privilege_abuse` is a measured blind spot.
9. **Do not** build any detector or rule on `geo_country`, `is_known_proxy_or_tor`,
   `is_internal_ip`, or `device_compliant_status` without reading **§8.1** — they are
   label proxies on the current datasets, and a rule using them wins for the wrong reason.
