# Skein — System Architecture

Skein is a dual-engine fraud detection platform built for **Razorpay AI Buildathon, Track 02 (AI Risk Manager)**. It combines an entity-level rolling statistical/ML velocity detector with a cross-account temporal graph ring detector, fusing both into an explainable per-transaction risk score.

---

## 1. End-to-End Pipeline Overview

```
                      [ Raw Transaction Stream ]
                     IEEE-CIS Transaction + Identity
                                    │
                                    ▼
                          [ inject_rings.py ]
                     40 Synthetic Fraud Rings (201 txns)
                                    │
                                    ▼
                      [ augmented_with_rings.csv ]
                           (590,741 rows)
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
      [ Engine 1: Velocity ]                [ Engine 2: Graph Ring ]
       phase2_velocity.py                    phase3_ring_detector.py
  • Past-only rolling counts            • Rare Device combos (< 55 txns)
    (1h, 1d, 7d)                        • 6-hour temporal co-occurrence
  • Amount Z-score & EWMA dev           • Louvain community detection (3-12)
  • Time deltas & entity fraud rate     • Oversized clique recovery (>=0.80, <=30m)
  • XGBoost (70/30 chronological split)        │
                 │                             ▼
                 ▼                    graph_clusters.json
       xgb_velocity_model.json                 │
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                      [ Engine 3: Fusion ]
                       phase4_fusion.py
             Risk = 0.50 * Velocity + 0.50 * Graph
             + Automated Plain-English Evidence Trail
                                │
                                ▼
                       fusion_scores.csv
                         (590,741 rows)
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
      [ Operational Policy ]         [ Live UI Dashboard ]
       phase5_evaluation.py                 app.py
    • Top 1% (Auto-Block: 94.6% prec)  • Streamlit interactive app
    • Top 5% (Review: 73.2% rec)       • NetworkX cluster visualizer
    • Top 10% (Aggressive)             • Risk tier policy badges
```

---

## 2. Core Components & Technical Specifications

### 2.1 Data Foundation & Ground Truth (`inject_rings.py`)
- **Base Dataset**: IEEE-CIS Fraud Detection (590,741 transactions across 13,637 unique `card1` entities).
- **Synthetic Rings Injected**: 40 rings (201 transactions total, ring size 3–7).
- **Link Mechanism**: Cloned `DeviceType + DeviceInfo` combinations from seed transactions with perturbed card attributes (`card1`, `card2`, `addr1`, `addr2`).
- **Temporal Burst**: Injected ring members execute transactions within 180 seconds of the seed transaction.
- **Ground Truth Labels**: `isFraud = 1`, `is_injected_ring = True`, `ring_id = [0..39]`.

### 2.2 Engine 1: Velocity & Statistical Anomaly Detector (`phase2_velocity.py`)
- **Principle**: Evaluates whether the current transaction deviates from the specific entity's (`card1`, `addr1`) historical baseline.
- **Chronological Split**: 70% Train (first 413,518 txns by `TransactionDT`) / 30% Test (last 177,223 txns). No random splitting; strictly zero data leakage from the future.
- **Engineered Past-Only Features (44 total)**:
  - Rolling transaction counts: 1 hour, 1 day, 7 days (`card1`); 1 hour, 1 day (`addr1`).
  - Cumulative transaction count & inter-transaction time delta (`time_since_last`).
  - Cumulative amount running mean & standard deviation.
  - Cumulative amount Z-score: $Z = (x - \mu_{\text{past}}) / \sigma_{\text{past}}$.
  - Amount Exponentially Weighted Moving Average (EWMA, $\alpha=2/6$) & absolute deviation from EWMA.
  - Cumulative historical fraud rate per entity.
  - Static numerical & categorical features: C1–C14, D1, D15, `ProductCD`, `card4`, `card6`, `TransactionAmt`, `log_amt`, `hour`, `day_of_week`.
- **Model**: `XGBClassifier` (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).
- **Artifact**: `xgb_velocity_model.json`.
- **Output**: `velocity_score` $\in [0.0, 1.0]$.
- **Test Performance (Held-out 30% split)**:
  - **XGBoost Velocity**: ROC-AUC = **0.8981**, PR-AUC = **0.5055**, Precision = **0.2507**, Recall = **0.6798**, F1 = **0.3663**.
  - **Z-Score / EWMA Baseline**: ROC-AUC = **0.5850**, PR-AUC = **0.0417**, Precision = **0.0404**, Recall = **0.1396**, F1 = **0.0627**.
  - *(Full dataset XGBoost: ROC-AUC = 0.9572, PR-AUC = 0.7410)*.

### 2.3 Engine 2: Temporal Graph Ring Detector (`phase3_ring_detector.py`)
- **Principle**: Detects multi-account fraud syndicates sharing rare technical fingerprints within tight temporal windows.
- **Graph Topology**:
  - **Nodes**: Unique `card1` entities (869 nodes in graph, collapses 590k transactions into entities).
  - **Edges**: 2,311 weighted undirected edges.
- **Filtering Rules & Hyperparameters**:
  1. **Rarity Threshold (`< 55`)**: Only device combinations (`DeviceType|DeviceInfo`) appearing in fewer than 55 transactions dataset-wide create edges. Eliminates mass-market device superclusters.
  2. **6-Hour Temporal Co-occurrence**: An edge between two cards is created *only* if their respective transactions on the shared rare device occurred within 6 hours ($|\Delta t| \le 21,600$ seconds).
  3. **Louvain Community Detection**: Partitions graph into communities. Clusters with size $3 \le N \le 12$ are flagged as candidate rings (51 clusters).
  4. **Oversized Community Recovery**: Communities with $> 12$ nodes (7 communities) are processed with maximal clique extraction:
     - **Density Filter**: Edge density $\ge 0.80$.
     - **Tight Temporal Burst Filter**: Sweep-line algorithm requires all clique card members to transact on the rare device within a **30-minute window** ($\le 1,800$ seconds).
     - **Cliques Recovered**: 35 dense, time-synchronized sub-clusters.
- **Total Clusters**: 86 (51 Louvain + 35 Recovered Cliques).
- **Artifact**: `graph_clusters.json` (maps 410 `card1` entities to cluster metadata).
- **Output**: `graph_score` $\in [0.0, 1.0]$ based on cluster size and internal edge density:
  $$\text{graph\_score} = \min\left(1.0, \frac{\text{size}}{7.0}\right) \times \text{density}$$

### 2.4 Engine 3: Fusion & Explainable Evidence (`phase4_fusion.py`)
- **Combined Risk Formula**:
  $$\text{final\_risk\_score} = 0.50 \times \text{velocity\_score} + 0.50 \times \text{graph\_score}$$
  *(Evaluated across $w \in [0.5, 0.9]$; $w=0.50$ achieves maximum test AUC of 0.9002 and best F1 of 0.5114).*
- **Automated Evidence Generation**: Evaluates velocity cutoff ($\ge 0.50$) and cluster membership to create plain-English audit strings:
  - *Dual Trigger*: `FLAGGED: velocity anomaly (score 0.93) + member of 9-card cluster (louvain) sharing rare 'mobile|SAMSUNG-SM-J320A Build/NMF26X'.`
  - *Graph Only*: `FLAGGED: member of 7-card cluster (louvain) sharing rare 'mobile|FIG-LX3 Build/HUAWEIFIG-LX3'.`
  - *Velocity Only*: `FLAGGED: velocity anomaly (score 0.97).`
  - *Clean*: `OK: velocity normal (0.19); no ring cluster.`
- **Artifact**: `fusion_scores.csv` (590,741 rows).

### 2.5 Operational Threshold Evaluation (`phase5_evaluation.py`)
Evaluates operational tiers at ₹50 per manual review:
- **Top 1% Cutoff ($\ge 0.4918$)**: 5,908 flagged | 5,587 TP | 321 FP | **94.6% precision** | **26.8% recall** | Review Cost ₹16,050 | Fraud Avoided ₹582,439.
- **Top 5% Cutoff ($\ge 0.3507$, Recommended)**: 29,538 flagged | 15,276 TP | 14,262 FP | **51.7% precision** | **73.2% recall** | Review Cost ₹713,100 | Fraud Avoided **₹2,196,790**.
- **Top 10% Cutoff ($\ge 0.2442$)**: 59,075 flagged | 17,899 TP | 41,176 FP | **30.3% precision** | **85.8% recall** | Review Cost ₹2,058,800 | Fraud Avoided ₹2,630,514.

### 2.6 Streamlit Investigation Dashboard (`app.py`)
- **Framework**: Streamlit 1.30+ with Matplotlib and NetworkX.
- **Key Modules**:
  - Executive KPI summary cards (Total volume, active cutoff, precision, recall, prevented fraud).
  - Operating tier radio selector (Top 1%, Top 5% [Recommended], Top 10%, Custom slider).
  - Sortable/filterable transaction queue with action policy badges (`🔴 Auto-Block`, `🟡 Review`, `🟢 Approve`).
  - Transaction deep-dive with progress risk gauge and plain-English audit box.
  - Interactive NetworkX entity graph visualizer highlighting the investigated `card1` against ring peers with shared device metadata.
  - 5 pre-loaded demo scenarios for live evaluation.

---

## 3. Metrics Distinction & Audited Coverage

To maintain rigorous transparency, we explicitly distinguish three distinct coverage statistics:

1. **Card Entity Coverage (3.01%)**:
   - `410` flagged `card1` entities out of `13,637` unique `card1` entities in the dataset (`410 / 13,637 = 3.0065%`).
2. **Cumulative Transaction Volume on Flagged Entities (27.85%)**:
   - `164,521` transactions out of `590,741` total dataset rows (`164,521 / 590,741 = 27.8499%`).
   - Driven by the power-law distribution of card transaction frequencies in IEEE-CIS, where a few high-velocity card IDs generate thousands of historical transactions.
3. **Specific Transactions on Cluster Device Fingerprint (0.30%)**:
   - `1,772` transactions out of `590,741` (`0.3000%` of all transactions, or `1.4913%` of transactions where device info is present).
   - Represents the precise transactions where the card entity actively used the shared rare device combo.

---

## 4. Key Results & Performance Summary

| Metric | Result | Context / Baseline |
|:---|:---:|:---|
| **Test ROC-AUC (Fused)** | **0.9002** | Evaluated on held-out last 30% chronological test split |
| **Test ROC-AUC (Velocity-Only)** | **0.8981** | XGBoost baseline on past-only features |
| **Full Ring Recall** | **92.5%** | 37 of 40 ground-truth rings fully unmasked |
| **Entity Precision** | **44.9%** | 184 TP / 410 flagged cards (13× dataset fraud rate of 3.53%) |
| **Entity Recall** | **91.5%** | 184 TP / 201 ground-truth ring cards identified |
| **Top 1% Ring Recall Lift** | **+79.1%** | Ring catch rate at Top 1% jumps from 7.0% (vel) to **86.1%** (fused) |
| **Top 5% Fraud Avoided** | **₹2,196,790** | 73.2% of all dataset fraud caught at 51.7% precision |

---

## 5. Explicit Limitations & Boundaries

1. **Rare Device Dependency**: The graph layer relies on `DeviceType + DeviceInfo` combinations appearing in `< 55` transactions. Fraud rings using mass-market devices (e.g., standard Windows/Chrome with counts > 50,000) are invisible to the graph and rely on Engine 1 velocity detection.
2. **Missing Device Metadata**: In the IEEE-CIS dataset, ~79.9% of transactions lack device identity information. The graph detector operates on the 20.1% of transactions with device data.
3. **Entity Precision (44.9%)**: ~55% of card entities in flagged clusters are coincidental users sharing a rare device rather than synthetic ring members. This is why the fusion weight is set to $0.50 / 0.50$.
4. **Synthetic Ground Truth**: The 40 fraud rings were synthetically injected to establish ground truth for multi-account syndicates. They should be evaluated as controlled benchmark injections rather than native dataset labels.
