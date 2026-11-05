import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

print("Loading data and computing exact Phase 2 metrics on 30% test split...")

# Run exact phase2 feature engineering
from phase2_velocity import add_entity_features, CAT_COLS, NUM_COLS

df = pd.read_csv("augmented_with_rings.csv")
df["hour"] = (df["TransactionDT"] // 3600) % 24
df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
df["log_amt"] = np.log1p(df["TransactionAmt"])

card1_windows = {"1h": 3600, "1d": 86400, "7d": 7 * 86400}
df = add_entity_features(df, "card1", card1_windows, prefix="card1")

addr1_windows = {"1h": 3600, "1d": 86400}
df = add_entity_features(df, "addr1", addr1_windows, prefix="addr1")

for c in CAT_COLS:
    df[c] = df[c].astype("category").cat.codes.astype("int32")

velocity_features = [
    "card1_tx_count_1h", "card1_tx_count_1d", "card1_tx_count_7d",
    "card1_cum_count", "card1_time_since_last",
    "card1_amt_cum_mean", "card1_amt_cum_std", "card1_amt_zscore",
    "card1_amt_ewma", "card1_amt_deviation", "card1_fraud_rate",
    "addr1_tx_count_1h", "addr1_tx_count_1d",
    "addr1_cum_count", "addr1_time_since_last",
    "addr1_amt_cum_mean", "addr1_amt_cum_std", "addr1_amt_zscore",
    "addr1_amt_ewma", "addr1_amt_deviation", "addr1_fraud_rate",
]
xgb_features = velocity_features + [
    "TransactionAmt", "log_amt", "hour", "day_of_week",
] + NUM_COLS + CAT_COLS

df.sort_values("TransactionDT", inplace=True)
df.reset_index(drop=True, inplace=True)

split_idx = int(len(df) * 0.70)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

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

# XGBoost evaluation (loading saved model)
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
print(f"  {'Metric':<16} {'Z-score/EWMA Baseline':>24} {'XGBoost Velocity':>24}")
print(f"  {'-'*16} {'-'*24} {'-'*24}")
print(f"  {'Precision':<16} {zscore_prec:>24.4f} {xgb_prec:>24.4f}")
print(f"  {'Recall':<16} {zscore_rec:>24.4f} {xgb_rec:>24.4f}")
print(f"  {'F1-Score':<16} {zscore_f1:>24.4f} {xgb_f1:>24.4f}")
print(f"  {'ROC-AUC':<16} {zscore_auc:>24.4f} {xgb_auc:>24.4f}")
print(f"  {'PR-AUC (Avg Prec)':<16} {zscore_prauc:>24.4f} {xgb_prauc:>24.4f}")
print("=" * 70)
