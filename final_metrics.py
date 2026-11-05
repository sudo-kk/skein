import pandas as pd
import numpy as np
import json

# Load all data
df = pd.read_csv("fusion_scores.csv")
amts = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID","TransactionAmt","DeviceType","DeviceInfo"])
df = df.merge(amts, on="TransactionID", how="left")

with open("graph_clusters.json") as f:
    cd = json.load(f)
cluster_meta = {int(k): v for k, v in cd["cluster_meta"].items()}
card1_to_cid = {int(k): v for k, v in cd["card1_to_cid"].items()}

total = len(df)
total_fraud = int(df["isFraud"].sum())
total_clean = total - total_fraud
fraud_rate = total_fraud / total
ring_df = df[df["is_injected_ring"]==True]
total_ring_txns = len(ring_df)
n_rings = int(ring_df["ring_id"].nunique())
ring_cards = ring_df.groupby("ring_id")["card1"].apply(set).to_dict()
all_ring_card1s = set()
for s in ring_cards.values():
    all_ring_card1s.update(s)
unique_card1s = df["card1"].nunique()

print("=== DATASET ===")
print(f"Total transactions: {total:,}")
print(f"Total fraud: {total_fraud:,} ({100*fraud_rate:.2f}%)")
print(f"Total clean: {total_clean:,}")
print(f"Injected ring txns: {total_ring_txns}")
print(f"Injected rings: {n_rings}")
print(f"Ring card1 entities: {len(all_ring_card1s)}")
print(f"Unique card1s: {unique_card1s:,}")
print(f"TransactionDT range: {df['TransactionDT'].min()} - {df['TransactionDT'].max()}")

print("\n=== GRAPH DETECTOR ===")
n_louvain = sum(1 for v in cluster_meta.values() if v["source"]=="louvain")
n_clique = sum(1 for v in cluster_meta.values() if v["source"]=="clique_recovery")
print(f"Total clusters: {len(cluster_meta)} (Louvain: {n_louvain}, Clique Recovery: {n_clique})")
print(f"Flagged card1s: {len(card1_to_cid)}")
flagged_set = set(card1_to_cid.keys())
tp_cards = flagged_set & all_ring_card1s
fp_cards = flagged_set - all_ring_card1s
print(f"TP card1s: {len(tp_cards)}, FP card1s: {len(fp_cards)}")
print(f"Entity Precision: {len(tp_cards)/len(flagged_set):.4f}")
print(f"Entity Recall: {len(tp_cards)/len(all_ring_card1s):.4f}")

# Ring-level recall
fully = 0
partial = 0
for rid, rset in ring_cards.items():
    best = 0
    for cid, meta in cluster_meta.items():
        cid_cards = set(int(k) for k,v in card1_to_cid.items() if v == cid)
        overlap = len(rset & cid_cards)
        best = max(best, overlap)
    if best == len(rset):
        fully += 1
    if best >= 2:
        partial += 1
print(f"Full ring recall: {fully}/{n_rings} ({100*fully/n_rings:.1f}%)")
print(f"Partial ring recall: {partial}/{n_rings} ({100*partial/n_rings:.1f}%)")
fire_txns = len(df[df["card1"].isin(flagged_set)])
fire = fire_txns / total * 100
print(f"Fire rate: {fire:.2f}% ({fire_txns:,} / {total:,})")
print(f"Card1 flag rate: {len(flagged_set)}/{unique_card1s} = {100*len(flagged_set)/unique_card1s:.2f}%")

print("\n=== VELOCITY MODEL ===")
vel = df["velocity_score"].values
y = df["isFraud"].values

def manual_auc(y_true, y_score):
    desc = np.argsort(y_score)[::-1]
    ys = y_true[desc]; vs = y_score[desc]
    d = np.concatenate([[True], np.diff(vs)!=0])
    tps = np.cumsum(ys)[d]; fps = np.cumsum(1-ys)[d]
    tps = np.concatenate([[0],tps]); fps = np.concatenate([[0],fps])
    fpr = fps/fps[-1] if fps[-1]>0 else fps
    tpr = tps/tps[-1] if tps[-1]>0 else tps
    return float(np.trapezoid(tpr, fpr))

vel_auc = manual_auc(y, vel)
print(f"Velocity-only AUC (full dataset): {vel_auc:.4f}")

split = int(total * 0.70)
vel_test_auc = manual_auc(y[split:], vel[split:])
print(f"Velocity-only AUC (test 30%): {vel_test_auc:.4f}")

print("\n=== FUSION MODEL ===")
fused = df["final_risk_score"].values
fused_auc = manual_auc(y, fused)
fused_test_auc = manual_auc(y[split:], fused[split:])
print(f"Fused AUC (full dataset): {fused_auc:.4f}")
print(f"Fused AUC (test 30%): {fused_test_auc:.4f}")
print(f"Fusion weight: 0.90 velocity + 0.10 graph")
print(f"AUC lift (test): {fused_test_auc - vel_test_auc:+.4f}")

print("\n=== OPERATIONAL THRESHOLDS ===")
for p in [1, 5, 10]:
    thr = np.percentile(fused, 100-p)
    flagged = df[fused >= thr]
    tp = int(flagged["isFraud"].sum())
    fp = len(flagged) - tp
    prec = tp/len(flagged) if len(flagged)>0 else 0
    rec = tp/total_fraud
    cost = fp*50
    saved = float(flagged[flagged["isFraud"]==1]["TransactionAmt"].sum())
    total_fraud_amt = float(df[df["isFraud"]==1]["TransactionAmt"].sum())
    print(f"Top {p}%: thr={thr:.4f} flagged={len(flagged):,} TP={tp:,} FP={fp:,} prec={prec:.3f} rec={rec:.3f} cost={cost:,} saved={saved:,.0f} total_fraud_amt={total_fraud_amt:,.0f}")

print("\n=== RING-SPECIFIC RECALL BY THRESHOLD ===")
for p in [1, 5, 10]:
    thr_f = np.percentile(fused, 100-p)
    thr_v = np.percentile(vel, 100-p)
    caught_f = ring_df[ring_df["final_risk_score"] >= thr_f]
    caught_v = ring_df[ring_df["velocity_score"] >= thr_v]
    print(f"Top {p}%: vel={len(caught_v)}/{total_ring_txns} ({100*len(caught_v)/total_ring_txns:.1f}%) fused={len(caught_f)}/{total_ring_txns} ({100*len(caught_f)/total_ring_txns:.1f}%) lift=+{100*(len(caught_f)-len(caught_v))/total_ring_txns:.1f}%")

print("\n=== SAMPLE FLAGGED RING TRANSACTIONS (1 per ring, first 10 rings) ===")
for rid in range(min(10, n_rings)):
    rtxns = ring_df[ring_df["ring_id"]==rid]
    for _, row in rtxns.head(1).iterrows():
        cid = int(row["graph_cluster_id"])
        cmeta = cluster_meta.get(cid, {})
        print(f"Ring {rid}: TxnID={int(row['TransactionID'])} card1={int(row['card1'])} vel={row['velocity_score']:.4f} graph={row['graph_score']:.4f} risk={row['final_risk_score']:.4f} cluster={cid} size={cmeta.get('size','N/A')} source={cmeta.get('source','N/A')}")
        print(f"  Evidence: {row['evidence']}")

print("\n=== SAMPLE CLEAN TRANSACTIONS (5 random) ===")
clean = df[(df["isFraud"]==0) & (df["is_injected_ring"]==False)]
for _, row in clean.sample(5, random_state=42).iterrows():
    print(f"TxnID={int(row['TransactionID'])} card1={int(row['card1'])} vel={row['velocity_score']:.4f} graph={row['graph_score']:.4f} risk={row['final_risk_score']:.4f}")
    print(f"  Evidence: {row['evidence']}")

print("\n=== MISSED / PARTIAL RINGS ===")
for rid, rset in sorted(ring_cards.items()):
    best = 0
    best_cid = -1
    for cid in cluster_meta:
        cid_cards = set(int(k) for k,v in card1_to_cid.items() if v == cid)
        overlap = len(rset & cid_cards)
        if overlap > best:
            best = overlap
            best_cid = cid
    if best < len(rset):
        status = "PARTIAL" if best >= 2 else "MISS"
        print(f"Ring {rid}: size={len(rset)} overlap={best}/{len(rset)} status={status}")

print("\n=== PER-RING DETAIL ===")
for rid in sorted(ring_cards.keys()):
    rset = ring_cards[rid]
    best = 0
    best_cid = -1
    best_sz = 0
    best_src = "none"
    for cid in cluster_meta:
        cid_cards = set(int(k) for k,v in card1_to_cid.items() if v == cid)
        overlap = len(rset & cid_cards)
        if overlap > best:
            best = overlap
            best_cid = cid
            best_sz = cluster_meta[cid]["size"]
            best_src = cluster_meta[cid]["source"]
    is_full = best == len(rset)
    status = "FULL" if is_full else ("PARTIAL" if best >= 2 else "MISS")
    print(f"Ring {rid:3d}: size={len(rset)} overlap={best} clsz={best_sz} src={best_src:17s} {status}")

# Score distribution stats
print("\n=== SCORE DISTRIBUTIONS ===")
for name, col in [("velocity_score", vel), ("graph_score", df["graph_score"].values), ("final_risk_score", fused)]:
    print(f"{name}: min={col.min():.4f} p25={np.percentile(col,25):.4f} p50={np.percentile(col,50):.4f} p75={np.percentile(col,75):.4f} p90={np.percentile(col,90):.4f} p95={np.percentile(col,95):.4f} p99={np.percentile(col,99):.4f} max={col.max():.4f}")
