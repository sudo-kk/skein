# Skein — Reproducibility Guide

This document provides exact, sequential instructions to reproduce all models, graphs, fused scores, evaluation tables, and the Streamlit dashboard from a clean environment.

---

## 1. Environment Setup

### Prerequisites
- Python `3.10`, `3.11`, `3.12`, or `3.13`
- Git
- Kaggle API credentials (if downloading raw IEEE-CIS dataset)

### Create Virtual Environment
```bash
# Clone repository
git clone https://github.com/sudo-kk/skein.git
cd skein

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Preparation & Ring Injection

### Step 2.1: Download IEEE-CIS Dataset
If `train_transaction.csv` and `train_identity.csv` are not already present in the workspace:
```bash
kaggle competitions download -c ieee-fraud-detection
unzip -o ieee-fraud-detection.zip train_transaction.csv train_identity.csv
```

### Step 2.2: Inject Ground-Truth Fraud Rings (Phase 1)
Inject 40 synthetic fraud rings (201 transactions) sharing rare device fingerprints:
```bash
python inject_rings.py \
    --transaction train_transaction.csv \
    --identity train_identity.csv \
    --out augmented_with_rings.csv \
    --n-rings 40 \
    --seed 42
```
**Expected Output**: `augmented_with_rings.csv` (590,741 rows, 435 columns).

---

## 3. End-to-End Execution Pipeline

Run each phase script in exact chronological order:

### Step 3.1: Train Velocity Detector (Phase 2)
Engineers past-only rolling features (1h/1d/7d tx counts, EWMA, amount Z-scores) on a chronological 70/30 split, evaluates against the unsupervised Z-score/EWMA baseline, and trains the XGBoost model:
```bash
python phase2_velocity.py
```
**Expected Outputs**:
- `xgb_velocity_model.json` (Trained model file, ~1.96 MB)
- **Held-Out Test (30%) Comparison Summary**:
  - **Z-score / EWMA Baseline**: Precision: **0.0404**, Recall: **0.1396**, F1: **0.0627**, ROC-AUC: **0.5850**, PR-AUC: **0.0417**
  - **XGBoost Velocity Model**: Precision: **0.2507**, Recall: **0.6798**, F1: **0.3663**, ROC-AUC: **0.8981**, PR-AUC: **0.5055**
- Full dataset XGBoost ROC-AUC: **0.9572**, PR-AUC: **0.7410**

---

### Step 3.2: Run Graph Ring Detector (Phase 3)
Builds the 6-hour temporal co-occurrence entity graph with rarity threshold `< 55`, executes Louvain community detection ($3 \le N \le 12$), and performs dense clique recovery ($\ge 0.80$ density, $\le 30$ min temporal burst) on oversized communities:
```bash
python phase3_ring_detector.py
```
**Expected Outputs**:
- `graph_clusters.json` (Cluster assignments for 410 `card1` entities across 86 clusters)
- Full Ring Recall: **92.5%** (37 / 40 rings)
- Entity Precision: **44.88%** (184 TP / 410 flagged cards)
- Entity Recall: **91.54%** (184 / 201 ring members)

---

### Step 3.3: Score Fusion & Evidence Generation (Phase 4)
Loads `xgb_velocity_model.json` and `graph_clusters.json`, calculates the 0.50 / 0.50 fused risk score, and generates plain-English audit strings for all 590,741 transactions:
```bash
python phase4_fusion.py
```
**Expected Outputs**:
- `fusion_scores.csv` (590,741 rows with `velocity_score`, `graph_score`, `final_risk_score`, `evidence`)
- Test ROC-AUC: **0.9002** (AUC Lift: **+0.0021**)

---

### Step 3.4: Operational Threshold Evaluation (Phase 5)
Evaluates the fused scores at Top 1%, Top 5%, and Top 10% risk thresholds:
```bash
python phase5_evaluation.py
```
**Expected Output Summary**:
```
==========================================================================================
  OPERATIONAL EVALUATION: FUSED RISK SCORE (final_risk_score)
==========================================================================================
Threshold  | Flagged  | TP     | FP     | Precision | Recall   | Review Cost (₹)| Fraud Avoided (₹)
------------------------------------------------------------------------------------------
Top 1%     | 5,908    | 5,587  | 321    | 94.6%     | 26.8%    | 16,050         | 582,439
Top 5%     | 29,538   | 15,276 | 14,262 | 51.7%     | 73.2%    | 713,100        | 2,196,790
Top 10%    | 59,075   | 17,899 | 41,176 | 30.3%     | 85.8%    | 2,058,800      | 2,630,514

================================================================================
  RING-SPECIFIC RECALL: VELOCITY vs. FUSED (Injected Rings Only)
================================================================================
Threshold  | Velocity-Only   | Fused Score     | Graph Incremental Lift
--------------------------------------------------------------------------------
Top 1%     | 7.0%            | 86.1%           | +79.1%
Top 5%     | 68.7%           | 98.5%           | +29.9%
Top 10%    | 76.1%           | 100.0%          | +23.9%
================================================================================
```

---

### Step 3.5: Comprehensive Metrics Verification
Run the unified verification script to print all verified numbers across all components:
```bash
python final_metrics.py
```

---

## 4. Launching the Interactive Dashboard (Phase 6)

Launch the Streamlit web application:
```bash
streamlit run app.py
```
Access the application in your browser at:
👉 `http://localhost:8501`

### Verifying Dashboard Features:
1. **Summary KPIs**: Check that Total Transactions (`590,741`), Top 5% Flagged (`29,538`), Precision (`51.7%`), and Recall (`73.2%`) display in the header.
2. **Threshold Switching**: Click between Top 1%, Top 5% (⭐ Recommended), and Top 10% to see dynamic queue updates.
3. **Demo Ring Case**: In the sidebar dropdown, select `🔥 Coordinated Ring Member (Txn 3577541, Ring #0)`:
   - Verify Policy shows `🔴 AUTO-BLOCK` with Risk Score `0.8000`.
   - Verify Evidence Trail shows: `FLAGGED: velocity anomaly (score 0.93) + member of 9-card cluster (louvain) sharing rare 'mobile|SAMSUNG-SM-J320A Build/NMF26X'.`
   - Verify the 9-node NetworkX cluster renders with the active card highlighted in red.
4. **Clean Case**: Select `✅ Clean Legitimate Transaction (Txn 3561399)`:
   - Verify Policy shows `🟢 AUTO-APPROVE` with Risk Score `0.1341`.
   - Verify Evidence Trail shows: `OK: velocity normal (0.27); no ring cluster.`

---

## 5. Artifact Checklist

Verify that the following generated files exist after running the pipeline:

| File | Expected Size | Description |
|:---|:---|:---|
| `augmented_with_rings.csv` | ~732 MB | Augmented dataset with 40 injected rings |
| `xgb_velocity_model.json` | ~1.96 MB | Trained XGBoost model |
| `graph_clusters.json` | ~15 KB | Detected cluster mappings and metadata |
| `fusion_scores.csv` | ~75 MB | Fused scores and plain-English evidence trails |
