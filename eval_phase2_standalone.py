import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import time

t0 = time.time()
print("Loading data...")

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

df = pd.read_csv("augmented_with_rings.csv", usecols=LOAD_COLS)
df.sort_values("TransactionDT", inplace=True)
df.reset_index(drop=True, inplace=True)

df["hour"] = (df["TransactionDT"] // 3600) % 24
df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
df["log_amt"] = np.log1p(df["TransactionAmt"])

def add_entity_features(df, entity_col, windows, prefix=None):
    pfx = prefix or entity_col
    n = len(df)
    sort_idx = df.sort_values([entity_col, "TransactionDT"]).index
    df = df.loc[sort_idx].reset_index(drop=True)

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

    mask = np.empty(n, dtype=bool)
    mask[0] = True
    mask[1:] = entity_vals[1:] != entity_vals[:-1]
    starts = np.nonzero(mask)[0]
    ends   = np.append(starts[1:], n)
    alpha = 2.0 / (5 + 1)

    for s, e in zip(starts, ends):
        g_times  = times[s:e]
        g_amts   = amts[s:e]
        g_frauds = frauds[s:e]
        ng = e - s

        for wn, wsec in windows.items():
            left = np.searchsorted(g_times, g_times - wsec, side="left")
            right = np.arange(ng)
            tx_counts[wn][s:e] = right - left

        cum_count[s:e] = np.arange(ng, dtype=np.float32)

        if ng > 1:
            time_gap[s + 1 : e] = np.diff(g_times).astype(np.float32)

        cs = np.cumsum(g_amts)
        cn = np.arange(1, ng + 1, dtype=np.float64)
        cmean = cs / cn
        css = np.cumsum(g_amts ** 2)
        cvar = np.maximum(css / cn - cmean ** 2, 0.0)
        cstd = np.sqrt(cvar)

        amt_mean[s] = g_amts[0]
        amt_std[s]  = 0.0
        if ng > 1:
            amt_mean[s + 1 : e] = cmean[: ng - 1].astype(np.float32)
            amt_std[s + 1 : e]  = cstd[: ng - 1].astype(np.float32)

        local_mean = amt_mean[s:e]
        local_std  = amt_std[s:e]
        safe_std = np.where(local_std > 0, local_std, 1.0)
        amt_zscore[s:e] = ((g_amts - local_mean) / safe_std).astype(np.float32)

        ew = np.empty(ng, dtype=np.float64)
        ew[0] = g_amts[0]
        for i in range(1, ng):
            ew[i] = alpha * g_amts[i] + (1.0 - alpha) * ew[i - 1]
        amt_ewma[s] = g_amts[0]
        if ng > 1:
            amt_ewma[s + 1 : e] = ew[: ng - 1].astype(np.float32)

        local_ewma = amt_ewma[s:e].astype(np.float64)
        safe_ewma = np.where(local_ewma > 0, local_ewma, 1.0)
        amt_dev[s:e] = (np.abs(g_amts - local_ewma) / safe_ewma).astype(np.float32)

        cf = np.cumsum(g_frauds)
        fraud_rate[s] = 0.0
        if ng > 1:
            fraud_rate[s + 1 : e] = (cf[: ng - 1] / np.arange(1, ng)).astype(np.float32)

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

    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

print("Engineering features...")
df = add_entity_features(df, "card1", {"1h": 3600, "1d": 86400, "7d": 604800}, prefix="card1")
df["_addr1"] = df["addr1"].fillna(-999).astype(int)
df = add_entity_features(df, "_addr1", {"1h": 3600, "1d": 86400}, prefix="addr1")
df.drop(columns=["_addr1"], inplace=True)

velocity_features = [c for c in df.columns if any(c.startswith(p) for p in ("card1_", "addr1_"))]
extra_numeric = [
    "TransactionAmt", "log_amt", "hour", "day_of_week",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "C10", "C11", "C12", "C13", "C14", "D1", "D15",
]

cat_cols = ["ProductCD", "card4", "card6"]
for c in cat_cols:
    le = LabelEncoder()
    df[c + "_enc"] = le.fit_transform(df[c].astype(str))

cat_encoded = [c + "_enc" for c in cat_cols]
xgb_features = velocity_features + extra_numeric + cat_encoded

split_idx = int(len(df) * 0.70)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

y_train = train["isFraud"].values
y_test  = test["isFraud"].values

# Baseline evaluation
def compute_anomaly_score(frame):
    score = np.zeros(len(frame))
    score += np.clip(frame["card1_tx_count_1h"].values, 0, 50) * 0.30
    score += np.clip(frame["card1_tx_count_1d"].values, 0, 100) * 0.10
    score += np.abs(frame["card1_amt_zscore"].values) * 0.25
    score += frame["card1_amt_deviation"].values * 0.15
    score += np.clip(frame["addr1_tx_count_1h"].values, 0, 50) * 0.15
    score += frame["card1_fraud_rate"].values * 0.05
    return score

train_scores = compute_anomaly_score(train)
test_scores  = compute_anomaly_score(test)

best_f1, best_thresh = 0.0, 0.0
for pct in np.arange(80, 99.9, 0.5):
    thr = np.percentile(train_scores, pct)
    preds = (train_scores >= thr).astype(int)
    f1 = f1_score(y_train, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thr

zscore_preds = (test_scores >= best_thresh).astype(int)
zscore_proba = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + 1e-9)

zscore_prec  = precision_score(y_test, zscore_preds, zero_division=0)
zscore_rec   = recall_score(y_test, zscore_preds, zero_division=0)
zscore_f1    = f1_score(y_test, zscore_preds, zero_division=0)
zscore_auc   = roc_auc_score(y_test, zscore_proba)
zscore_prauc = average_precision_score(y_test, zscore_proba)

# Load existing saved XGBoost model
model = xgb.XGBClassifier()
model.load_model("xgb_velocity_model.json")

X_test = test[xgb_features].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

xgb_preds = model.predict(X_test)
xgb_proba = model.predict_proba(X_test)[:, 1]

xgb_prec  = precision_score(y_test, xgb_preds, zero_division=0)
xgb_rec   = recall_score(y_test, xgb_preds, zero_division=0)
xgb_f1    = f1_score(y_test, xgb_preds, zero_division=0)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_prauc = average_precision_score(y_test, xgb_proba)

print("\n" + "=" * 70)
print("  VERIFIED PHASE 2 EVALUATION METRICS (Held-out 30% Test Set)")
print("=" * 70)
print(f"  {'Metric':<18} {'Z-score/EWMA Baseline':>22} {'XGBoost Velocity':>22}")
print(f"  {'-'*18} {'-'*22} {'-'*22}")
print(f"  {'Precision':<18} {zscore_prec:>22.4f} {xgb_prec:>22.4f}")
print(f"  {'Recall':<18} {zscore_rec:>22.4f} {xgb_rec:>22.4f}")
print(f"  {'F1-Score':<18} {zscore_f1:>22.4f} {xgb_f1:>22.4f}")
print(f"  {'ROC-AUC':<18} {zscore_auc:>22.4f} {xgb_auc:>22.4f}")
print(f"  {'PR-AUC':<18} {zscore_prauc:>22.4f} {xgb_prauc:>22.4f}")
print("=" * 70)
