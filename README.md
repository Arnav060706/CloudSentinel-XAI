# CloudSentinel-XAI

**AI-Driven Zero-Trust Security Framework for Explainable Threat Detection in Multi-Cloud Environments**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%2B%20Isolation%20Forest-orange.svg)](#-ai-and-ml-approach)
[![XAI](https://img.shields.io/badge/XAI-SHAP-8E44AD.svg)](https://shap.readthedocs.io/)
[![Grafana](https://img.shields.io/badge/Visualization-Grafana-F46800.svg)](https://grafana.com/)
[![Prometheus](https://img.shields.io/badge/Metrics-Prometheus-E6522C.svg)](https://prometheus.io/)
[![Loki](https://img.shields.io/badge/Logging-Grafana%20Loki-F2CC0C.svg)](https://grafana.com/oss/loki/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://sqlite.org/)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic-E92063.svg)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#-license)

---

CloudSentinel-XAI ingests **AWS CloudTrail**, **Azure AD Audit / Sign-in**, and **GCP Cloud Audit** logs, stitches identity across clouds, scores risk in real time with parallel ML models, and explains every critical alert with SHAP-attributed, faithfulness-gated narratives from a local LLM — built for SOC analysts who need to *trust* an explanation, not just receive one.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Why CloudSentinel-XAI](#-why-cloudsentinel-xai)
- [Features](#-features)
- [System Architecture](#️-system-architecture)
- [How It Works](#-how-it-works)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Running the Project](#-running-the-project)
- [Usage / API](#-usage--api)
- [Synthetic Dataset Generator](#-synthetic-dataset-generator)
- [Testing](#-testing)
- [Monitoring & Dashboards](#-monitoring--dashboards)
- [Project Status & Roadmap](#-project-status--roadmap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Overview

CloudSentinel-XAI is a high-throughput, multi-cloud Zero-Trust security framework that:

1. **Ingests** raw or pre-normalized audit logs from AWS, Azure, and GCP into a single common event schema.
2. **Stitches identity** across clouds using a tiered correlation model (federation join → creation provenance → fuzzy fusion), so the same human or service account is tracked as one entity even when cloud-native identifiers don't line up.
3. **Scores risk** in parallel using an unsupervised anomaly detector (Isolation Forest) and a supervised ATT&CK-phase classifier (XGBoost), combined with a time-decayed, cross-cloud-aware risk score.
4. **Explains** every critical alert through TreeSHAP feature attribution, gated by a **faithfulness test** (a SHAP deletion test) so a local LLM only narrates explanations that are provably grounded in the model's own behavior — not hallucinated.
5. **Surfaces** everything through Prometheus metrics, Loki-backed logs, and Grafana dashboards for real-time SOC visibility.

It is designed for Security Operations Centers (SOCs) and cloud security teams that need interpretable, real-time, cross-cloud threat detection without sacrificing detection accuracy — and without asking analysts to trust a black box.

---

## 🎯 Why CloudSentinel-XAI

Most cloud-native detection tools operate **per-cloud** and treat explanation as an afterthought. CloudSentinel-XAI is built around three specific, deliberately-engineered ideas:

- **Pace-independent cross-cloud detection.** Identity and an entity's lifetime cloud footprint are tracked independently of the live risk-scoring window, so a low-and-slow attacker pacing steps minutes apart across AWS → Azure → GCP is still caught, not just an attacker who moves fast.
- **Faithfulness-gated explanations.** SHAP attributions are only allowed to reach the LLM narrator after passing a deletion test — zeroing the top-attributed features and confirming the model's confidence actually drops. If it doesn't, the alert is flagged for manual analyst review instead of generating a plausible-sounding but unfaithful story.
- **Leakage-safe synthetic evaluation.** The included dataset generator deliberately avoids the most common synthetic-IAM-dataset flaw (identity as a label giveaway) by compromising randomly-drawn *legitimate* users and holding out a generalization split — so reported detection numbers mean something.

---

## ✨ Features

- 🔍 Multi-cloud audit log ingestion (AWS CloudTrail, Azure AD Audit/Sign-in, GCP Cloud Audit Logs) — raw or pre-normalized
- ⚡ High-speed, schema-unifying log normalization pipeline with GeoIP/ASN and Tor exit-node enrichment
- 🧩 Tiered, cross-cloud identity stitching (federation join → creation provenance → fuzzy fusion with ambiguity guard)
- 🧠 Parallel ML inference — Isolation Forest anomaly scoring + XGBoost MITRE ATT&CK phase classification
- 📈 Time-decayed, cross-cloud-diversity-aware risk scoring with automation-vs-human sensitivity
- 📊 SHAP-based explainable AI with per-alert top feature attribution
- 🛡️ Faithfulness-gated local LLM narratives (Llama 3.2 via Ollama) — no narrative without a passing deletion test
- 🧪 Leakage-safe, MITRE-ATT&CK-annotated synthetic multi-cloud dataset generator for training and evaluation
- 📡 Prometheus metrics, 📝 Loki centralized logging, 📈 Grafana real-time dashboards
- 💾 SQLite-backed alert and risk-state persistence with write-coalescing background flush
- 🧯 Graceful **bypass mode** — the entire pipeline runs end-to-end with neutral defaults even before trained models exist, so the system is never a single missing artifact away from crashing

---

## 🏗️ System Architecture

![CloudSentinel-XAI Architecture](assets/Architecture.png)

At a glance, telemetry flows through five layers:

| Layer | Responsibility |
|---|---|
| **Multi-Cloud Control Plane** | AWS CloudTrail, Azure AD Audit, GCP Cloud Audit Logs ingestion |
| **Normalization Layer** | Common event schema (actor, action, resource, timestamp, source signals) |
| **Feature Engineering** | Lexical, identity-context, network-context, and behavioral features |
| **ML + XAI Inference** | Isolation Forest + XGBoost, TreeSHAP attribution, faithfulness gate |
| **Identity Stitching + Risk Scoring** | Tiered cross-cloud correlation, time-decayed cross-cloud risk score |
| **Forensic Dashboard** | Faithfulness-gated LLM narrative, Loki/Prometheus/Grafana |

---

## 🔬 How It Works

1. **Ingest** — `POST /api/v1/ingest/raw` accepts raw provider JSON and normalizes it through the parser pipeline; `POST /api/v1/ingest/raw` accepts already-normalized records directly.
2. **Enrich & Normalize** — Each event is parsed into a `UnifiedLogModel`, enriched with GeoLite2 ASN/country lookups and Tor exit-node matching, and reduced to a shared feature set.
3. **Identity Resolution** — `MultiCloudGraphEngine` resolves the event to a persistent entity via federation join → creation-provenance lookup → fuzzy fusion (UA family/version, proxy/Tor alignment, principal type, capped IP match with an ambiguity margin to avoid over-merging).
4. **Parallel ML Inference** — `ParallelMLEngine` concurrently runs Isolation Forest (anomaly score) and XGBoost (ATT&CK phase + TreeSHAP attribution) via `asyncio.to_thread`.
5. **Risk Scoring** — `RiskEngine` combines a time-decayed sum of anomaly scores with a cross-cloud diversity multiplier, driven by the entity's *lifetime* cloud footprint (not just the live window), and distinguishes automation from human principals.
6. **Faithfulness Gate + Narrative** — On a critical alert, `FaithfulnessGatedXAI` re-runs the model with top-SHAP features zeroed out; only if confidence drops enough does it prompt a local Llama 3.2 model (via Ollama) for a two-sentence tactical narrative, with the raw telemetry explicitly delimited to reduce prompt-injection risk.
7. **Persistence & Observability** — Risk state is write-coalesced into SQLite, narratives are pushed to Loki, and Prometheus metrics feed a Grafana dashboard for live SOC monitoring.
8. **Bypass Mode** — If no trained model artifacts are found under `models/`, every stage above still runs, returning well-formed neutral defaults instead of failing — so the service is deployable and testable before training is complete.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Backend / API | FastAPI, Pydantic, Uvicorn |
| Machine Learning | XGBoost, Isolation Forest, scikit-learn |
| Explainability | SHAP (TreeExplainer + deletion-test faithfulness gate) |
| LLM Narration | Llama 3.2 via Ollama (local inference) |
| Database | SQLite (async, via SQLAlchemy) |
| Monitoring | Prometheus |
| Logging | Grafana Loki |
| Visualization | Grafana |
| Enrichment | MaxMind GeoLite2 (ASN + Country), Tor exit-node list |
| Language | Python 3.12 |

---

## 📂 Project Structure

```
CloudSentinel-XAI/
│
├── app/
│   ├── core/                    # Database engine/session setup
│   ├── parser_normalizer/       # AWS / Azure / GCP parsers, enrichment, feature extraction
│   │   ├── mock_data/           # Sample raw logs per cloud + a unified sample stream
│   │   ├── reference_data/      # GeoLite2 DBs, Tor exit-node list
│   │   └── src/                 # ParserPipeline, normalizer, schema, feature_extractor
│   ├── routers/                 # /api/v1/ingest, /api/v1/ingest/raw
│   ├── services/                # graph_engine, risk_engine, ml_inference, xai_engine,
│   │                             # xai_triage, db_flusher, metrics_exporter, loki_exporter
│   └── main.py                  # FastAPI app, lifespan, engine bootstrap, bypass-mode logic
│
├── dataset_script/              # Leakage-safe synthetic multi-cloud dataset generator
│   ├── env_profile.py           # Procedural, seeded, label-neutral organization/population
│   ├── emitters.py              # Emits records in exact AWS/Azure/GCP schemas
│   ├── generate_benign.py       # Benign baseline traffic
│   ├── attack_scenarios.py      # MITRE ATT&CK cross-cloud kill chains
│   ├── generate_attacks.py      # Renders attacks on randomly-drawn legitimate victims
│   ├── check_leakage.py         # Asserts no identity is a label giveaway
│   ├── validate_realism.py      # Synthetic-vs-real statistical fidelity report
│   └── build_dataset.py         # One command → full labelled corpus
│
├── demo/                        # Stage-by-stage demo scripts (parser → feature → graph → ML → risk → XAI → persistence)
├── tests/                       # pytest suite (end-to-end + unit, runs in bypass mode)
├── assets/
│   └── Architecture.png
│
├── grafana.json                 # Grafana dashboard definition
├── loki-config.yaml             # Loki configuration
├── prometheus.yml                # Prometheus scrape configuration
├── metrics_server.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Arnav060706/CloudSentinel-XAI.git
cd CloudSentinel-XAI

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Optional, for local LLM narration:** install [Ollama](https://ollama.com/) and pull the model used by the XAI engine:

```bash
ollama pull llama3.2
```

---

## 🚀 Running the Project

```bash
uvicorn app.main:app --reload
```

| Resource | URL |
|---|---|
| API base | `http://127.0.0.1:8000` |
| Interactive docs (Swagger UI) | `http://127.0.0.1:8000/docs` |
| Health check | `http://127.0.0.1:8000/health` |
| Prometheus metrics | `http://127.0.0.1:8000/metrics` |

On startup, the app will:
- Create SQLite tables if they don't exist.
- Attempt to load trained artifacts from `models/` (`isolation_forest.pkl`, `xgboost_model.json`, `feature_encoder.pkl`, `class_names.json`).
- Fall back to **bypass mode** automatically if any are missing, logging a clear warning — the API remains fully usable, and dropping trained artifacts into `models/` later upgrades scoring without any code changes.

---

## 📡 Usage / API

### Ingest raw multi-cloud logs

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest/raw \
  -H "Content-Type: application/json" \
  -d '[
        {
          "eventVersion": "1.08",
          "userIdentity": {
            "type": "IAMUser",
            "principalId": "AIDAEXAMPLE001",
            "arn": "arn:aws:iam::112233445566:user/alice.chen",
            "accountId": "112233445566",
            "userName": "alice.chen"
          },
          "eventTime": "2026-06-10T08:14:32Z",
          "eventSource": "signin.amazonaws.com",
          "eventName": "ConsoleLogin",
          "awsRegion": "us-east-1",
          "sourceIPAddress": "203.0.113.45",
          "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
          "responseElements": { "ConsoleLogin": "Success" }
        }
      ]'
```

Response:

```json
{
  "status": "Telemetry Accepted",
  "records_normalized": 1,
  "records_failed": 0
}
```

### Ingest pre-normalized logs

```bash
POST /api/v1/ingest
Content-Type: application/json

[ { "...": "UnifiedLogModel-shaped record" } ]
```

Both endpoints return `202 Accepted` immediately and process each event asynchronously through: parallel ML inference → identity stitching → risk scoring → faithfulness-gated narrative (on critical alerts) → persistence.

---

## 🧪 Synthetic Dataset Generator

CloudSentinel-XAI ships a research-grade synthetic multi-cloud IAM dataset generator, purpose-built to avoid the label-leakage problems common in synthetic security datasets.

```bash
cd dataset_script

python build_dataset.py       # -> dataset/ (~12k events, ~2.7% malicious, 17 ATT&CK techniques)
python check_leakage.py       # verifies no identity is a label giveaway
python validate_realism.py    # synthetic-vs-real statistical fidelity report
```

Highlights:
- Procedurally generated organization (default 300 users, 25 services, ~630 IPs including Tor/hosting/foreign) over 14 days, calibrated to real observed log shapes.
- Attacks compromise **randomly-drawn legitimate users** — no "villain" identities — so every principal appears both benign and (sometimes) attacked.
- A held-out generalization split (~15% of identities) tests detection on principals never seen in training.
- 17 MITRE ATT&CK techniques across 9 tactics, spanning 4 cross-cloud kill chains.
- Configurable attack pacing (`--pace fast|slow|mixed`) to test detection against low-and-slow evasion specifically.

See [`dataset_script/readme.md`](dataset_script/readme.md) for full generator documentation, reproducibility flags, and the research roadmap this dataset is designed to support.

---

## ✅ Testing

```bash
PYTHONPATH=. pytest -q
```

The test suite runs entirely in **bypass mode** (no trained models required), validating:
- End-to-end ingestion → normalization → persistence through the live FastAPI app
- Lifetime cross-cloud footprint tracking in the identity graph engine
- Risk-engine behavior under single-cloud vs. multi-cloud, and baseline-suppressed legitimate multi-cloud identities (e.g. DevOps accounts)

---

## 📊 Monitoring & Dashboards

- **Prometheus** (`prometheus.yml`) scrapes `/metrics` for per-cloud ML scores, risk intensity, cloud-span counts, and critical-alert counters.
- **Loki** (`loki-config.yaml`) receives structured log lines for every critical alert, including LLM narratives (or a structured fallback summary if generation was gated or failed).
- **Grafana** (`grafana.json`) visualizes:
  - Real-time incident monitoring
  - Threat severity and risk trends
  - MITRE ATT&CK phase mapping
  - XAI narrative / SHAP driver panels
  - Alert timelines


---


Developed as part of the **CloudSentinel-XAI CCNCS Research Project**.

[⬆ Back to Top](#cloudsentinel-xai)