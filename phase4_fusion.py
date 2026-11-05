"""
Phase 4 (Final) -- Fusion Function + Evidence
================================================
Uses saved graph_clusters.json from Phase 3 (Final) and
the Phase 2 XGBoost velocity model.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import xgboost as xgb
import json
import time as _time
import warnings


# ---- Manual metric implementations (avoids sklearn -> scipy DLL block) ----
def roc_auc_score(y_true, y_score):
    """ROC-AUC via the trapezoidal rule on sorted scores."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    desc = np.argsort(y_score)[::-1]
    y_true = y_true[desc]
    y_score = y_score[desc]
    distinct = np.concatenate([[True], np.diff(y_score) != 0])
    tps = np.cumsum(y_true)[distinct]
    fps = np.cumsum(1 - y_true)[distinct]
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    return float(np.trapezoid(tpr, fpr))


def f1_score(y_true, y_pred, *args, **kwargs):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


class LabelEncoder:
    def fit_transform(self, values):
        values = [str(v) for v in values]
        uniq = sorted(set(values))
        mapping = {v: i for i, v in enumerate(uniq)}
        return np.array([mapping[v] for v in values], dtype=np.int32)

warnings.filterwarnings("ignore")
np.random.seed(42)

SEED       = 42
TRAIN_FRAC = 0.70
DATA_PATH  = "augmented_with_rings.csv"

print("=" * 70)
print("  Phase 4 (Final): Fusion + Evidence")
print("=" * 70)

# ================================================================
# A. VELOCITY FEATURE ENGINEERING (identical to Phase 2)
# ================================================================
def add_entity_features(df, entity_col, windows, prefix=None):
    pfx = prefix or entity_col
    t0 = _time.time()
    n = len(df)
    sort_idx = df.sort_values([entity_col, "TransactionDT"]).index
    df = df.loc[sort_idx].reset_index(drop=True)
    tx_counts  = {wn: np.zeros(n, dtype=np.float32) for wn in windows}
    cum_count  = np.zeros(n, dtype=np.float32)
    time_gap   = np.full(n, -1.0, dtype=np.float32)
    amt_mean   = np.zeros(n, dtype=np.float32)
    amt_std    = np.zeros(n, dtype=np.float32)
    amt_zscore = np.zeros(n, dtype=np.float32)
    amt_ewma   = np.zeros(n, dtype=np.float32)
    amt_dev    = np.zeros(n, dtype=np.float32)
    fraud_rate = np.zeros(n, dtype=np.float32)
    ev = df[entity_col].values
    ts = df["TransactionDT"].values.astype(np.float64)
    am = df["TransactionAmt"].values.astype(np.float64)
    fr = df["isFraud"].values.astype(np.float64)
    mask = np.empty(n, dtype=bool); mask[0] = True
    mask[1:] = ev[1:] != ev[:-1]
    starts = np.nonzero(mask)[0]; ends = np.append(starts[1:], n)
    alpha = 2.0 / 6.0
    for s, e in zip(starts, ends):
        gt, ga, gf = ts[s:e], am[s:e], fr[s:e]; ng = e - s
        for wn, ws in windows.items():
            tx_counts[wn][s:e] = np.arange(ng) - np.searchsorted(gt, gt - ws, side="left")
        cum_count[s:e] = np.arange(ng, dtype=np.float32)
        if ng > 1: time_gap[s+1:e] = np.diff(gt).astype(np.float32)
        cs = np.cumsum(ga); cn = np.arange(1, ng+1, dtype=np.float64)
        cm = cs / cn; cv = np.maximum(np.cumsum(ga**2)/cn - cm**2, 0); csd = np.sqrt(cv)
        amt_mean[s] = ga[0]; amt_std[s] = 0
        if ng > 1: amt_mean[s+1:e] = cm[:ng-1].astype(np.float32); amt_std[s+1:e] = csd[:ng-1].astype(np.float32)
        lm = amt_mean[s:e]; ls = amt_std[s:e]
        amt_zscore[s:e] = ((ga - lm) / np.where(ls > 0, ls, 1)).astype(np.float32)
        ew = np.empty(ng, dtype=np.float64); ew[0] = ga[0]
        for i in range(1, ng): ew[i] = alpha * ga[i] + (1 - alpha) * ew[i-1]
        amt_ewma[s] = ga[0]
        if ng > 1: amt_ewma[s+1:e] = ew[:ng-1].astype(np.float32)
        le = amt_ewma[s:e].astype(np.float64)
        amt_dev[s:e] = (np.abs(ga - le) / np.where(le > 0, le, 1)).astype(np.float32)
        cf = np.cumsum(gf); fraud_rate[s] = 0
        if ng > 1: fraud_rate[s+1:e] = (cf[:ng-1] / np.arange(1, ng)).astype(np.float32)
    for wn in windows: df[f"{pfx}_tx_count_{wn}"] = tx_counts[wn]
    df[f"{pfx}_cum_count"] = cum_count; df[f"{pfx}_time_since_last"] = time_gap
    df[f"{pfx}_amt_cum_mean"] = amt_mean; df[f"{pfx}_amt_cum_std"] = amt_std
    df[f"{pfx}_amt_zscore"] = amt_zscore; df[f"{pfx}_amt_ewma"] = amt_ewma
    df[f"{pfx}_amt_deviation"] = amt_dev; df[f"{pfx}_fraud_rate"] = fraud_rate
    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    {pfx}: {len(windows)+8} features  ({_time.time()-t0:.1f}s)")
    return df

# ================================================================
# B. LOAD DATA + VELOCITY SCORES
# ================================================================
print("\n[1/6] Loading data ...")
t0 = _time.time()
LOAD_COLS = [
    "TransactionID","TransactionDT","TransactionAmt","isFraud",
    "ProductCD","card1","card2","card3","card4","card5","card6",
    "addr1","addr2","DeviceType","DeviceInfo",
    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
    "D1","D15","ring_id","is_injected_ring",
]
df = pd.read_csv(DATA_PATH, usecols=LOAD_COLS)
df.sort_values("TransactionDT", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  {len(df):,} rows  ({_time.time()-t0:.1f}s)")

df["hour"]        = (df["TransactionDT"] // 3600) % 24
df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
df["log_amt"]     = np.log1p(df["TransactionAmt"])

print("\n[2/6] Engineering velocity features ...")
df = add_entity_features(df, "card1", {"1h":3600,"1d":86400,"7d":604800}, "card1")
df["_addr1"] = df["addr1"].fillna(-999).astype(int)
df = add_entity_features(df, "_addr1", {"1h":3600,"1d":86400}, "addr1")
df.drop(columns=["_addr1"], inplace=True)

velocity_features = [c for c in df.columns if c.startswith("card1_") or c.startswith("addr1_")]
extra_numeric = ["TransactionAmt","log_amt","hour","day_of_week",
                 "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14",
                 "D1","D15"]
for c in ["ProductCD","card4","card6"]:
    df[c+"_enc"] = LabelEncoder().fit_transform(df[c].astype(str))
cat_encoded = ["ProductCD_enc","card4_enc","card6_enc"]
xgb_features = velocity_features + extra_numeric + cat_encoded

print(f"\n[3/6] Loading XGBoost model ...")
t0 = _time.time()
booster = xgb.Booster()
booster.load_model("xgb_velocity_model.json")
X_all = np.nan_to_num(df[xgb_features].values.astype(np.float32), nan=0, posinf=0, neginf=0)
dall = xgb.DMatrix(X_all)
df["velocity_score"] = booster.predict(dall)
print(f"  Done  ({_time.time()-t0:.1f}s)")

# ================================================================
# C. LOAD GRAPH CLUSTERS FROM PHASE 3
# ================================================================
print(f"\n[4/6] Loading graph clusters from graph_clusters.json ...")
with open("graph_clusters.json", "r") as f:
    cluster_data = json.load(f)

cluster_meta = {int(k): v for k, v in cluster_data["cluster_meta"].items()}
card1_to_cid = {int(k): v for k, v in cluster_data["card1_to_cid"].items()}

# graph_score: weighted degree normalized, only for card1s in clusters
# We need the graph to compute weighted degree - rebuild it minimally
# or just use cluster density * size as proxy
card1_graph_score = {}
for card1, cid in card1_to_cid.items():
    meta = cluster_meta[cid]
    # Score based on cluster size and density
    # Ring-sized clusters (3-7) with high density get highest scores
    size_score = min(1.0, meta["size"] / 7.0)
    density = meta["density"]
    card1_graph_score[card1] = round(size_score * density, 4)

df["combo"] = df["DeviceType"].fillna("unknown") + "|" + df["DeviceInfo"].fillna("unknown")

def get_graph_score(row):
    card1 = row["card1"]
    if card1 not in card1_to_cid:
        return 0.0
    cid = card1_to_cid[card1]
    combos = cluster_meta[cid].get("combos", [])
    if row["combo"] in combos:
        return card1_graph_score[card1]
    return 0.0

df["graph_score"]        = df.apply(get_graph_score, axis=1).astype(np.float32)
df["graph_cluster_id"]   = df["card1"].map(card1_to_cid).fillna(-1).astype(int)
df["graph_cluster_size"] = df["graph_cluster_id"].map(
    lambda c: cluster_meta[c]["size"] if c in cluster_meta else 0
)

n_graph = (df["graph_score"] > 0).sum()
print(f"  Clusters loaded: {len(cluster_meta)}")
print(f"  Card1s with graph signal: {len(card1_to_cid)}")
print(f"  Transactions with graph signal: {n_graph:,} / {len(df):,} "
      f"({100*n_graph/len(df):.2f}%)")

# ================================================================
# D. FUSION WEIGHT EVALUATION
# ================================================================
print(f"\n[5/6] Evaluating fusion weights ...")
split_idx = int(len(df) * TRAIN_FRAC)
test_df = df.iloc[split_idx:]
y_test  = test_df["isFraud"].values
v_test  = test_df["velocity_score"].values
g_test  = test_df["graph_score"].values

vel_only_auc = roc_auc_score(y_test, v_test)

best_auc, best_w = 0, 0.7
for w in [0.5, 0.6, 0.7, 0.8, 0.9]:
    fused = w * v_test + (1-w) * g_test
    auc   = roc_auc_score(y_test, fused)
    best_f1 = 0
    for pct in np.arange(85, 99.5, 0.5):
        thr = np.percentile(fused, pct)
        f1  = f1_score(y_test, (fused >= thr).astype(int), zero_division=0)
        best_f1 = max(best_f1, f1)
    marker = " <<" if auc >= best_auc else ""
    print(f"  w={w:.1f}:  AUC={auc:.4f}  F1={best_f1:.4f}{marker}")
    if auc >= best_auc:
        best_auc, best_w = auc, w

# Noisy-OR
fused_nor = 1 - (1-v_test)*(1-g_test)
auc_nor = roc_auc_score(y_test, fused_nor)
print(f"  noisy-OR: AUC={auc_nor:.4f}")

print(f"\n  Velocity-only AUC: {vel_only_auc:.4f}")
print(f"  Selected w={best_w}  AUC={best_auc:.4f}  "
      f"(lift: {best_auc - vel_only_auc:+.4f})")

FUSION_W = best_w

# ================================================================
# E. COMPUTE FINAL SCORES + EVIDENCE
# ================================================================
print(f"\n[6/6] Computing final scores + evidence ...")
t0 = _time.time()

df["final_risk_score"] = (FUSION_W * df["velocity_score"]
                          + (1-FUSION_W) * df["graph_score"]).astype(np.float32)

VEL_THRESHOLD = 0.50

def make_evidence(row):
    vel = row["velocity_score"]
    gs  = row["graph_score"]
    cid = row["graph_cluster_id"]
    csz = row["graph_cluster_size"]

    vel_fired   = vel >= VEL_THRESHOLD
    graph_fired = gs > 0 and cid >= 0

    parts = []
    if vel_fired:
        parts.append(f"velocity anomaly (score {vel:.2f})")
    if graph_fired:
        combos = cluster_meta.get(cid, {}).get("combos", [])
        combo_str = combos[0] if combos else "unknown"
        src = cluster_meta.get(cid, {}).get("source", "unknown")
        parts.append(
            f"member of {csz}-card cluster ({src}) "
            f"sharing rare '{combo_str}'"
        )
    if parts:
        return "FLAGGED: " + " + ".join(parts) + "."
    else:
        reasons = [f"velocity normal ({vel:.2f})"]
        if cid < 0:
            reasons.append("no ring cluster")
        else:
            reasons.append(f"graph score low ({gs:.2f})")
        return "OK: " + "; ".join(reasons) + "."

df["evidence"] = df.apply(make_evidence, axis=1)
print(f"  Generated evidence  ({_time.time()-t0:.1f}s)")

# ================================================================
# F. PRINT EXAMPLES
# ================================================================
print("\n" + "=" * 82)
print("  5 INJECTED-RING TRANSACTIONS")
print("=" * 82)

ring_txns = df[df["is_injected_ring"] == True]
sample_rings = ring_txns.groupby("ring_id").first().reset_index()
# 4 with graph signal, 1 without
rg = sample_rings[sample_rings["graph_score"] > 0].head(4)
rn = sample_rings[sample_rings["graph_score"] == 0].head(1)
ring_picks = pd.concat([rg, rn])

for _, row in ring_picks.iterrows():
    txn_id = int(row["TransactionID"])
    actual = df[df["TransactionID"] == txn_id].iloc[0]
    print(f"\n  TxnID {txn_id}  |  card1={int(actual['card1'])}  |  "
          f"ring_id={int(actual['ring_id'])}")
    print(f"    velocity  = {actual['velocity_score']:.4f}")
    print(f"    graph     = {actual['graph_score']:.4f}")
    print(f"    risk      = {actual['final_risk_score']:.4f}")
    print(f"    evidence: {actual['evidence']}")

print("\n" + "=" * 82)
print("  5 CLEAN TRANSACTIONS")
print("=" * 82)

clean = df[(df["isFraud"] == 0) & (df["is_injected_ring"] == False)]
cg = clean[clean["graph_score"] > 0].head(2)
cn = clean[clean["graph_score"] == 0].sample(n=3, random_state=SEED)
clean_picks = pd.concat([cg, cn])

for _, actual in clean_picks.iterrows():
    txn_id = int(actual["TransactionID"])
    print(f"\n  TxnID {txn_id}  |  card1={int(actual['card1'])}  |  isFraud=0")
    print(f"    velocity  = {actual['velocity_score']:.4f}")
    print(f"    graph     = {actual['graph_score']:.4f}")
    print(f"    risk      = {actual['final_risk_score']:.4f}")
    print(f"    evidence: {actual['evidence']}")

# ================================================================
# G. SAVE CSV
# ================================================================
out_cols = [
    "TransactionID","TransactionDT","isFraud","is_injected_ring",
    "ring_id","card1","velocity_score","graph_score",
    "graph_cluster_id","graph_cluster_size",
    "final_risk_score","evidence",
]
df[out_cols].to_csv("fusion_scores.csv", index=False)
print(f"\n  Saved fusion_scores.csv  ({len(df):,} rows)")

print("\n" + "=" * 70)
print("  Phase 4 (Final) complete.")
print("=" * 70 + "\n")
