# 🛡️ Skein: Fraud Ring & Velocity Risk Manager

[![Track](https://img.shields.io/badge/Track-02%20AI%20Risk%20Manager-6C5CE7?style=flat-square)](https://github.com/sudo-kk/skein)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-blue?style=flat-square)](https://python.org)
[![Test ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.9002-success?style=flat-square)](#-key-results)
[![Ring Recall](https://img.shields.io/badge/Full%20Ring%20Recall-92.5%25-success?style=flat-square)](#-key-results)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-red?style=flat-square)](#-interactive-dashboard-phase-6)

> **Submission for Razorpay AI Buildathon — Track 02: AI Risk Manager**  
> An explainable, dual-engine fraud defense platform combining **entity-level past-only rolling velocity models** with a **temporal graph ring detector** to unmask coordinated multi-account fraud syndicates.

---

## 📌 Problem Statement

Modern payment fraud has evolved from isolated, single-account anomalies into **organized fraud rings (syndicates)**:
1. **Coordinated Multi-Account Attacks**: Fraudsters distribute attacks across dozens of distinct credit card accounts (`card1`) and billing addresses (`addr1`).
2. **Velocity Evasion**: By executing only 1–2 transactions per card with intervals of several hours, fraudsters keep per-entity velocity below standard rule triggers.
3. **The Hidden Flaw**: Because a single syndicate operates the attack, all accounts secretly share low-level technical infrastructure (device emulators, rare OS builds, browser fingerprints) and transact within synchronized time windows.

Standard machine learning models evaluate transactions **one card at a time**, allowing coordinated rings to cash out undetected. Naive graph models that link all shared devices drown in false positives by creating giant superclusters around popular hardware (e.g., standard iPhones or Windows PCs).

---

## 💡 Solution Overview

**Skein** introduces a defense-only, multi-layered risk management architecture:
- **Engine 1 (Micro-Level Velocity)**: Evaluates whether an individual account deviates from its own historical baseline using strictly past-only rolling statistical features and an XGBoost classifier.
- **Engine 2 (Macro-Level Temporal Graph)**: Constructs an entity-level co-occurrence graph restricted to **rare device fingerprints (`< 55` transactions)** and **6-hour temporal windows**, isolating dense rings via Louvain community detection ($3 \le N \le 12$) and **tight temporal clique recovery** ($\ge 0.80$ density, $\le 30$ min burst window).
- **Engine 3 (Fusion & Explainability)**: Blends both probabilities ($0.50 \times \text{Velocity} + 0.50 \times \text{Graph}$) into a unified $0.0–1.0$ risk score and automatically generates plain-English audit trails explaining *why* a transaction was flagged.

---

## 🏛️ System Architecture

```
                       [ Incoming Transaction Stream ]
                       590,741 Transactions (IEEE-CIS)
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
   [ ENGINE 1: ROLLING VELOCITY ]            [ ENGINE 2: TEMPORAL GRAPH ]
   • Past-only rolling counts (1h, 1d, 7d)   • Rare device filter (< 55 txns)
   • Amount Z-scores & EWMA deviations       • 6-hour temporal co-occurrence
   • Cumulative entity fraud rate            • Louvain community detection (3-12)
   • XGBoost (70/30 chronological split)     • Dense clique recovery (>=0.80, <=30m)
                 │                                         │
                 ▼                                         ▼
       velocity_score [0, 1]                     graph_score [0, 1]
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
                        [ ENGINE 3: FUSION & AUDIT ]
                     final_risk = 0.50*Vel + 0.50*Graph
                     + Automated Plain-English Evidence Trail
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
     [ OPERATIONAL POLICY ]                      [ LIVE DASHBOARD ]
  • Top 1% (Auto-Block: 94.6% prec)           • Streamlit Interactive App
  • Top 5% (Manual Review: 73.2% rec)         • NetworkX Cluster Visualizer
  • Top 10% (Aggressive Review)               • Policy Tier Status Badges
```

---

## 📊 Dataset & Ring-Injection Methodology (Phase 1)

- **Source Dataset**: IEEE-CIS Fraud Detection dataset (590,741 transactions, 13,637 unique `card1` entities, ~182 days of activity).
- **Synthetic Ground Truth Rings**: Injected **40 realistic fraud rings** (201 transactions total, ring sizes 3–7) via [`inject_rings.py`](inject_rings.py).
- **Syndicate Simulation**: Each ring was seeded from a real transaction. Ring members received perturbed card identifiers (`card1`, `card2`, `addr1`) but inherited the seed's rare `DeviceType + DeviceInfo` combination, transacting within a 180-second window. All injected ring transactions were marked `isFraud = 1` and `is_injected_ring = True`.

---

## 🔬 Core Components Breakdown

### 1. Phase 2 — Velocity / Spike Detector
- **Feature Engineering**: 44 features engineered strictly from past transactions (zero future leakage):
  - Rolling transaction counts over 1 hour, 1 day, and 7 days.
  - Inter-transaction time intervals (`time_since_last`).
  - Running amount mean, standard deviation, cumulative Z-scores, and EWMA ($\alpha=2/6$) deviations.
  - Cumulative historical entity fraud rates.
- **Model**: Evaluated against an unsupervised Z-score/EWMA statistical baseline and trained an XGBoost classifier on a **strict chronological 70% train / 30% test split** (`TransactionDT`).
- **Verified Held-Out Performance (Test 30%)**:

| Metric | Z-Score / EWMA Baseline | XGBoost Velocity Model |
|:---|:---:|:---:|
| **Precision** | 4.04% (0.0404) | **25.07% (0.2507)** |
| **Recall** | 13.96% (0.1396) | **67.98% (0.6798)** |
| **F1-Score** | 0.0627 | **0.3663** |
| **ROC-AUC** | 0.5850 | **0.8981** |
| **PR-AUC** | 0.0417 | **0.5055** |

*(Full dataset XGBoost ROC-AUC: 0.9572, PR-AUC: 0.7410)*

### 2. Phase 3 — Graph Ring Detector
- **Entity Graph**: Collapses 590k transactions into 869 `card1` nodes and 2,311 edges.
- **Configuration & Sensitivity Analysis**:
  - **Rarity Threshold `< 55`**: Captures rings whose shared device counts range from 50–53 without creating superclusters around mass-market devices.
  - **6-Hour Temporal Window**: An edge between two cards is created only if they used the rare device within 6 hours of each other.
  - **Louvain Community Detection**: Detects 51 ring-sized clusters ($3 \le N \le 12$).
  - **Oversized Community Recovery**: For communities with $> 12$ nodes (7 communities), a sweep-line algorithm extracts sub-cliques with **edge density $\ge 0.80$** and a **$\le 30$-minute burst window**, recovering 35 tight sub-clusters and discarding 116 ambient coincidences.
- **Detector Performance**: **92.5% Full Ring Recall** (37 / 40 rings), **44.88% Entity Precision** (184 TP / 410 flagged cards), and **91.54% Entity Recall**.

### 3. Phase 4 — Fusion & Explainability
- **Optimal Weighting**: Evaluated weights $w \in [0.5, 0.9]$. Selected **$0.50 \times \text{Velocity} + 0.50 \times \text{Graph}$**, achieving test ROC-AUC of **0.9002** (lift of **+0.0021** over velocity alone).
- **Plain-English Evidence Trail**: Every transaction receives a natural-language audit trail:
  - *Dual Trigger*: `FLAGGED: velocity anomaly (score 0.93) + member of 9-card cluster (louvain) sharing rare 'mobile|SAMSUNG-SM-J320A Build/NMF26X'.`
  - *Graph Only*: `FLAGGED: member of 7-card cluster (louvain) sharing rare 'mobile|FIG-LX3 Build/HUAWEIFIG-LX3'.`
  - *Clean*: `OK: velocity normal (0.19); no ring cluster.`

### 4. Phase 5 — Operational Threshold Evaluation
Evaluated against a manual review cost of **₹50 per false positive**:

| Policy Tier | Risk Cutoff | Flagged Volume | Precision | Fraud Recall | Review Cost | Fraud Loss Avoided |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Top 1% (Auto-Block)** | $\ge 0.4918$ | 5,908 (1.0%) | **94.6%** | 26.8% | ₹16,050 | ₹582,439 |
| **Top 5% (⭐ Recommended)** | $\ge 0.3507$ | 29,538 (5.0%) | **51.7%** | **73.2%** | ₹713,100 | **₹2,196,790** |
| **Top 10% (Aggressive)** | $\ge 0.2442$ | 59,075 (10.0%) | **30.3%** | 85.8% | ₹2,058,800 | ₹2,630,514 |

#### Ring-Specific Recall Lift (Injected Rings Only):
- **Top 1% Tier**: Velocity alone catches only 7.0% of ring members. Fused score catches **86.1%** (**+79.1% incremental lift**).
- **Top 5% Tier**: Velocity catches 68.7% → Fused score catches **98.5%** (**+29.9% lift**).
- **Top 10% Tier**: Velocity catches 76.1% → Fused score catches **100.0%** (**+23.9% lift**).

---

## 📈 Key Results Summary

```
========================================================================================
  SKEIN FINAL VERIFICATION SCORECARD
========================================================================================
  Chronological Test ROC-AUC (Fused):       0.9002  (+0.0021 lift over velocity-only)
  Chronological Test ROC-AUC (Velocity):    0.8981
  Full Ring Recall (Ground Truth):          92.50%  (37 / 40 rings completely unmasked)
  Entity-Level Ring Recall:                 91.54%  (184 / 201 ring members caught)
  Entity-Level Precision:                   44.88%  (184 TP / 410 flagged cards, 13x base)
  Top 1% Ring Recall Lift:                  +79.1%  (7.0% -> 86.1% in auto-block tier)
  Top 5% Est. Fraud Loss Avoided:              ₹2,196,790 (73.2% of total dataset fraud)
  Top 1% Auto-Block Precision:              94.6%   (5,587 TP / 5,908 flagged)
========================================================================================
```

---

## 🔍 Audited Coverage Statistics

To ensure scientific rigor, we explicitly distinguish three distinct coverage statistics:
1. **Flagged Card Entities (3.01%)**: `410` unique `card1` accounts flagged out of `13,637` total unique cards in the dataset (`410 / 13,637 = 3.0065%`).
2. **Cumulative Transactions on Flagged Entities (27.85%)**: `164,521` total rows belong to those 410 flagged entities (`164,521 / 590,741 = 27.8499%`), driven by the power-law frequency of high-volume merchant/payroll cards in IEEE-CIS.
3. **Specific Transactions on Cluster Device Fingerprints (0.30%)**: `1,772` transactions actively used the shared rare device combination (`1,772 / 590,741 = 0.3000%` of all transactions, or `1.4913%` of transactions with device info present).

---

## ⚠️ Honest Limitations & Boundaries

1. **Dependence on Rare Device Fingerprints**: The graph layer connects accounts via rare `DeviceType + DeviceInfo` combinations (`< 55` occurrences). Fraud rings operating from common mass-market hardware (e.g., standard Windows/Chrome with counts > 50,000) are invisible to the graph and rely on Engine 1 velocity detection.
2. **Missing Device Metadata**: In IEEE-CIS, ~79.9% of transactions lack device identity data. The graph operates on the 20.1% of transactions where device information is present.
3. **Entity Precision (44.9%)**: ~55% of card entities in flagged clusters are coincidental users sharing rare hardware rather than fraudsters. This is why fusion weights the graph at 50% rather than exclusively relying on it.
4. **Synthetic Ground Truth**: The 40 fraud rings were synthetically injected to benchmark multi-account coordination. They should be evaluated as controlled benchmark injections rather than native labels.

---

## 💻 Interactive Dashboard (Phase 6)

Launch the Streamlit dashboard to explore transactions, inspect policy tiers, and visualize entity clusters:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

**Dashboard Features**:
- **Executive KPIs**: Real-time stats on transaction volume, cutoff thresholds, precision, recall, and preventable losses.
- **Interactive Policy Switcher**: Toggle between Top 1%, Top 5% (⭐ Recommended), Top 10%, or custom percentiles.
- **Transaction Queue Table**: Filter by flagged queue, ground-truth rings, or graph clusters with action badges (`🔴 Auto-Block`, `🟡 Review`, `🟢 Approve`).
- **Entity Graph Visualizer**: Renders interactive NetworkX cluster graphs showing the investigated card against its peers with shared device metadata.
- **Quick-Pick Demo Dropdown**: Jump directly to 5 preloaded scenarios (Coordinated Ring Member, Recovered Clique, Pure Velocity Spike, Clean User).

---

## 🚀 Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/sudo-kk/skein.git
cd skein
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Get the Data
The dataset is **not** in this repo: IEEE-CIS is Kaggle competition data that can't be redistributed (and the CSVs are ~700 MB). Accept the competition rules on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data), set up your [Kaggle API token](https://www.kaggle.com/docs/api), then:
```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip train_transaction.csv train_identity.csv
```
Every other artifact (`augmented_with_rings.csv`, `xgb_velocity_model.json`, `graph_clusters.json`, `fusion_scores.csv`) is regenerated by the pipeline below.

### 3. Run Pipeline from Scratch
```bash
# 1. Inject 40 synthetic fraud rings (Phase 1)
python inject_rings.py --transaction train_transaction.csv --identity train_identity.csv --out augmented_with_rings.csv --n-rings 40

# 2. Train Velocity Detector (Phase 2)
python phase2_velocity.py

# 3. Build & evaluate Graph Ring Detector (Phase 3)
python phase3_ring_detector.py

# 4. Fuse scores and generate plain-English evidence (Phase 4)
python phase4_fusion.py

# 5. Evaluate operational thresholds & costs (Phase 5)
python phase5_evaluation.py

# 6. Verify full metrics scorecard
python final_metrics.py

# 7. Launch Streamlit Dashboard (Phase 6)
streamlit run app.py
```

For full reproduction details and step-by-step artifact checklists, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).  
For detailed architectural specifications and data flow diagrams, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 🗂️ Project Layout

| Path | What it does |
|:---|:---|
| `inject_rings.py` | Phase 1: seeds 40 synthetic fraud rings into IEEE-CIS |
| `phase2_velocity.py` | Phase 2: past-only rolling features + XGBoost velocity model |
| `phase3_ring_detector.py`, `phase3_sensitivity.py` | Phase 3: rare-device temporal graph, Louvain + clique recovery, threshold sweeps |
| `phase4_fusion.py` | Phase 4: score fusion and plain-English evidence trails |
| `phase5_evaluation.py`, `final_metrics.py` | Phase 5: policy tiers, review cost, final scorecard |
| `app.py` | Phase 6: Streamlit dashboard |
| `audit_*.py`, `trace_*.py`, `debug_rings.py`, `eval_phase2_*.py` | Audits and investigations used to verify the numbers above |

---

## 📄 License & Attribution
Built by [Karthik V K](https://github.com/sudo-kk) for **Razorpay AI Buildathon 2026** (Track 02: AI Risk Manager). Defense-only implementation.

Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Vesta Corporation via Kaggle), used under the competition's terms and not redistributed here.
