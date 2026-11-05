"""
Phase 2 -- Velocity / Spike Detector
=====================================
Engineers rolling velocity features per entity, trains:
  1. Z-score/EWMA statistical baseline (unsupervised anomaly scoring)
  2. XGBoost classifier (supervised)

Both evaluated on a time-based 70/30 train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    classification_report, f1_score, average_precision_score,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import time as _time
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

# ================================================================
# CONFIG
# ================================================================
DATA_PATH = "augmented_with_rings.csv"
TRAIN_FRAC = 0.70
SEED = 42

# Columns to load (keeps memory reasonable on the 700 MB CSV)
LOAD_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C12", "C13", "C14",
    "D1", "D15",
    "ring_id", "is_injected_ring",
]

# Rolling-count time windows (seconds)
WINDOWS = {"1h": 3600, "1d": 86400, "7d": 604800}

print("=" * 62)
print("  Phase 2: Velocity / Spike Detector")
print("=" * 62)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n[1/7] Loading data ...")
t0 = _time.time()
df = pd.read_csv(DATA_PATH, usecols=LOAD_COLS)
df.sort_values("TransactionDT", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Loaded {len(df):,} rows x {df.shape[1]} cols  ({_time.time()-t0:.1f}s)")

# ================================================================
# 2. BASIC TIME FEATURES
# ================================================================
print("\n[2/7] Adding time features ...")
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
df["log_amt"] = np.log1p(df["TransactionAmt"])

# ================================================================
# 3. PER-ENTITY VELOCITY FEATURES
# ================================================================
def add_entity_features(df, entity_col, windows, prefix=None):
    """Compute rolling/cumulative velocity features per entity.

    For each transaction by entity *e*, all features are computed
    using only *past* transactions by *e* (shifted by 1 row inside
    each group) so there is no look-ahead leakage.
    """
    pfx = prefix or entity_col
    t0 = _time.time()
    n = len(df)

    # Pre-sort by entity + time so each group is chronological
    sort_idx = df.sort_values([entity_col, "TransactionDT"]).index
    df = df.loc[sort_idx].reset_index(drop=True)

    # Pre-allocate numpy arrays for speed
    tx_counts = {wn: np.zeros(n, dtype=np.float32) for wn in windows}
    cum_count    = np.zeros(n, dtype=np.float32)
    time_gap     = np.full(n, -1.0, dtype=np.float32)
    amt_mean     = np.zeros(n, dtype=np.float32)
    amt_std      = np.zeros(n, dtype=np.float32)
    amt_zscore   = np.zeros(n, dtype=np.float32)
    amt_ewma     = np.zeros(n, dtype=np.float32)
    amt_dev      = np.zeros(n, dtype=np.float32)
    fraud_rate   = np.zeros(n, dtype=np.float32)

    entity_vals = df[entity_col].values
    times       = df["TransactionDT"].values.astype(np.float64)
    amts        = df["TransactionAmt"].values.astype(np.float64)
    frauds      = df["isFraud"].values.astype(np.float64)

    # Locate group boundaries (entity is sorted)
    mask = np.empty(n, dtype=bool)
    mask[0] = True
    mask[1:] = entity_vals[1:] != entity_vals[:-1]
    starts = np.nonzero(mask)[0]
    ends   = np.append(starts[1:], n)

    alpha = 2.0 / (5 + 1)  # EWMA span=5

    for s, e in zip(starts, ends):
        g_times  = times[s:e]
        g_amts   = amts[s:e]
        g_frauds = frauds[s:e]
        ng = e - s

        # --- rolling transaction counts ---
        for wn, wsec in windows.items():
            left = np.searchsorted(g_times, g_times - wsec, side="left")
            right = np.arange(ng)
            tx_counts[wn][s:e] = right - left   # excludes current row

        # --- cumulative count (past tx by this entity) ---
        cum_count[s:e] = np.arange(ng, dtype=np.float32)

        # --- time since last transaction ---
        if ng > 1:
            time_gap[s + 1 : e] = np.diff(g_times).astype(np.float32)

        # --- cumulative mean/std of amount (shifted: only past) ---
        cs = np.cumsum(g_amts)
        cn = np.arange(1, ng + 1, dtype=np.float64)
        cmean = cs / cn
        css = np.cumsum(g_amts ** 2)
        cvar = np.maximum(css / cn - cmean ** 2, 0.0)
        cstd = np.sqrt(cvar)
        # Shift by 1: for tx i, use stats from tx 0..i-1
        amt_mean[s] = g_amts[0]
        amt_std[s]  = 0.0
        if ng > 1:
            amt_mean[s + 1 : e] = cmean[: ng - 1].astype(np.float32)
            amt_std[s + 1 : e]  = cstd[: ng - 1].astype(np.float32)

        # --- z-score of current amount vs past stats ---
        local_mean = amt_mean[s:e]
        local_std  = amt_std[s:e]
        safe_std = np.where(local_std > 0, local_std, 1.0)
        amt_zscore[s:e] = ((g_amts - local_mean) / safe_std).astype(np.float32)

        # --- EWMA of amount (shifted) ---
        ew = np.empty(ng, dtype=np.float64)
        ew[0] = g_amts[0]
        for i in range(1, ng):
            ew[i] = alpha * g_amts[i] + (1.0 - alpha) * ew[i - 1]
        amt_ewma[s] = g_amts[0]
        if ng > 1:
            amt_ewma[s + 1 : e] = ew[: ng - 1].astype(np.float32)

        # --- amount deviation from EWMA ---
        local_ewma = amt_ewma[s:e].astype(np.float64)
        safe_ewma = np.where(local_ewma > 0, local_ewma, 1.0)
        amt_dev[s:e] = (np.abs(g_amts - local_ewma) / safe_ewma).astype(np.float32)

        # --- cumulative fraud rate (shifted: only past labels) ---
        cf = np.cumsum(g_frauds)
        fraud_rate[s] = 0.0
        if ng > 1:
            fraud_rate[s + 1 : e] = (cf[: ng - 1] / np.arange(1, ng)).astype(np.float32)

    # Write columns
    for wn in windows:
        df[f"{pfx}_tx_count_{wn}"] = tx_counts[wn]
    df[f"{pfx}_cum_count"]      = cum_count
    df[f"{pfx}_time_since_last"] = time_gap
    df[f"{pfx}_amt_cum_mean"]   = amt_mean
    df[f"{pfx}_amt_cum_std"]    = amt_std
    df[f"{pfx}_amt_zscore"]     = amt_zscore
    df[f"{pfx}_amt_ewma"]       = amt_ewma
    df[f"{pfx}_amt_deviation"]  = amt_dev
    df[f"{pfx}_fraud_rate"]     = fraud_rate

    # Re-sort chronologically
    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)

    elapsed = _time.time() - t0
    print(f"  {pfx}: {len(windows)} windows, 11 features  ({elapsed:.1f}s)")
    return df


print("\n[3/7] Engineering entity velocity features ...")

# Primary entity: card1 (present for every transaction)
df = add_entity_features(df, "card1", WINDOWS, prefix="card1")

# Secondary entity: addr1 (has NaNs -- fill with sentinel so groupby works)
df["_addr1"] = df["addr1"].fillna(-999).astype(int)
df = add_entity_features(df, "_addr1", {"1h": 3600, "1d": 86400}, prefix="addr1")
df.drop(columns=["_addr1"], inplace=True)

print(f"  Total columns after feature engineering: {df.shape[1]}")

# ================================================================
# 4. PREPARE FEATURE MATRIX
# ================================================================
print("\n[4/7] Building feature matrix & time-based split ...")

# Velocity features (used by both models)
velocity_features = [c for c in df.columns if any(
    c.startswith(p) for p in ("card1_", "addr1_")
)]

# Extra features for XGBoost (raw counts + categoricals)
extra_numeric = [
    "TransactionAmt", "log_amt", "hour", "day_of_week",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C12", "C13", "C14", "D1", "D15",
]

# Label-encode categoricals for XGBoost
cat_cols = ["ProductCD", "card4", "card6"]
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c + "_enc"] = le.fit_transform(df[c].astype(str))
    le_dict[c] = le

cat_encoded = [c + "_enc" for c in cat_cols]

xgb_features = velocity_features + extra_numeric + cat_encoded

# --- Time-based split ---
split_idx = int(len(df) * TRAIN_FRAC)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

y_train = train["isFraud"].values
y_test  = test["isFraud"].values

print(f"  Velocity features:  {len(velocity_features)}")
print(f"  XGBoost features:   {len(xgb_features)}")
print(f"  Train: {len(train):,} rows  (fraud rate {y_train.mean():.4f})")
print(f"  Test:  {len(test):,}  rows  (fraud rate {y_test.mean():.4f})")
print(f"  Split time boundary: TransactionDT = {train['TransactionDT'].iloc[-1]:,}")

# ================================================================
# 5. MODEL A: Z-SCORE / EWMA STATISTICAL BASELINE
# ================================================================
print("\n[5/7] Training Z-score / EWMA baseline ...")
t0 = _time.time()

# Composite anomaly score: weighted sum of absolute z-scores and
# deviation signals.  The weights are hand-tuned heuristics (not
# learned from labels -- this is the "unsupervised" baseline).
def compute_anomaly_score(frame):
    score = np.zeros(len(frame))
    # Velocity spikes (high tx count in short window = suspicious)
    score += np.clip(frame["card1_tx_count_1h"].values, 0, 50) * 0.30
    score += np.clip(frame["card1_tx_count_1d"].values, 0, 100) * 0.10
    # Amount anomaly
    score += np.abs(frame["card1_amt_zscore"].values) * 0.25
    score += frame["card1_amt_deviation"].values * 0.15
    # Address velocity
    score += np.clip(frame["addr1_tx_count_1h"].values, 0, 50) * 0.15
    # Historical fraud rate of this entity
    score += frame["card1_fraud_rate"].values * 0.05
    return score


train_scores = compute_anomaly_score(train)
test_scores  = compute_anomaly_score(test)

# Tune threshold on training set (maximize F1)
best_f1, best_thresh = 0.0, 0.0
for pct in np.arange(80, 99.9, 0.5):
    thr = np.percentile(train_scores, pct)
    preds = (train_scores >= thr).astype(int)
    f1 = f1_score(y_train, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thr

print(f"  Best threshold (F1={best_f1:.4f} on train): {best_thresh:.4f}")

# Evaluate on TEST
zscore_preds = (test_scores >= best_thresh).astype(int)
zscore_proba = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-9)

zscore_prec  = precision_score(y_test, zscore_preds, zero_division=0)
zscore_rec   = recall_score(y_test, zscore_preds, zero_division=0)
zscore_auc   = roc_auc_score(y_test, zscore_proba)
zscore_prauc = average_precision_score(y_test, zscore_proba)
zscore_f1    = f1_score(y_test, zscore_preds, zero_division=0)

print(f"  Z-score/EWMA on TEST ({_time.time()-t0:.1f}s):")
print(f"    Precision : {zscore_prec:.4f}")
print(f"    Recall    : {zscore_rec:.4f}")
print(f"    F1        : {zscore_f1:.4f}")
print(f"    ROC-AUC   : {zscore_auc:.4f}")
print(f"    PR-AUC    : {zscore_prauc:.4f}")

# ================================================================
# 6. MODEL B: XGBOOST CLASSIFIER
# ================================================================
print("\n[6/7] Training XGBoost classifier ...")
t0 = _time.time()

X_train = train[xgb_features].values.astype(np.float32)
X_test  = test[xgb_features].values.astype(np.float32)

# Replace any remaining NaN / inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

# Class imbalance: scale_pos_weight
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / max(pos, 1)
print(f"  scale_pos_weight = {spw:.2f}  (neg={neg:,}, pos={pos:,})")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=spw,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

xgb_preds = model.predict(X_test)
xgb_proba = model.predict_proba(X_test)[:, 1]

xgb_prec  = precision_score(y_test, xgb_preds, zero_division=0)
xgb_rec   = recall_score(y_test, xgb_preds, zero_division=0)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_prauc = average_precision_score(y_test, xgb_proba)
xgb_f1    = f1_score(y_test, xgb_preds, zero_division=0)

print(f"  XGBoost on TEST ({_time.time()-t0:.1f}s):")
print(f"    Precision : {xgb_prec:.4f}")
print(f"    Recall    : {xgb_rec:.4f}")
print(f"    F1        : {xgb_f1:.4f}")
print(f"    ROC-AUC   : {xgb_auc:.4f}")
print(f"    PR-AUC    : {xgb_prauc:.4f}")

# ================================================================
# 7. RESULTS COMPARISON
# ================================================================
print("\n" + "=" * 62)
print("  RESULTS SUMMARY  (test set, time-based split)")
print("=" * 62)
print(f"  {'Metric':<14} {'Z-score/EWMA':>14} {'XGBoost':>14}")
print(f"  {'-'*14} {'-'*14} {'-'*14}")
print(f"  {'Precision':<14} {zscore_prec:>14.4f} {xgb_prec:>14.4f}")
print(f"  {'Recall':<14} {zscore_rec:>14.4f} {xgb_rec:>14.4f}")
print(f"  {'F1':<14} {zscore_f1:>14.4f} {xgb_f1:>14.4f}")
print(f"  {'ROC-AUC':<14} {zscore_auc:>14.4f} {xgb_auc:>14.4f}")
print(f"  {'PR-AUC':<14} {zscore_prauc:>14.4f} {xgb_prauc:>14.4f}")
print("=" * 62)

# Top-10 XGBoost feature importances
print("\n  Top-10 XGBoost Feature Importances:")
imp = model.feature_importances_
top10 = np.argsort(imp)[::-1][:10]
for rank, idx in enumerate(top10, 1):
    print(f"    {rank:2d}. {xgb_features[idx]:<30s}  {imp[idx]:.4f}")

# --- Classification report (XGBoost, detailed) ---
print("\n  XGBoost Classification Report (test set):")
print(classification_report(y_test, xgb_preds, target_names=["Legit", "Fraud"], digits=4))

# --- Save the trained model ---
model.save_model("xgb_velocity_model.json")
print("  Saved XGBoost model -> xgb_velocity_model.json")
print("  Done.\n")
