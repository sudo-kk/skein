import pandas as pd
import numpy as np

# Load scores and amounts
print("Loading data...")
df = pd.read_csv("fusion_scores.csv")
df_amts = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "TransactionAmt"])

# Merge amounts
df = df.merge(df_amts, on="TransactionID", how="left")

# Total fraud and rings
total_txns = len(df)
total_fraud = df["isFraud"].sum()
injected_ring_txns = df[df["is_injected_ring"] == True]
total_injected = len(injected_ring_txns)

print(f"Total Transactions: {total_txns:,}")
print(f"Total Fraud: {total_fraud:,}")
print(f"Total Injected Ring Txns: {total_injected:,}\n")

percentiles = [1, 5, 10]
results = []
ring_results = []

for p in percentiles:
    # Fused Risk Thresholds
    threshold_risk = np.percentile(df["final_risk_score"], 100 - p)
    flagged_risk = df[df["final_risk_score"] >= threshold_risk]
    
    tp_risk = flagged_risk["isFraud"].sum()
    fp_risk = len(flagged_risk) - tp_risk
    prec_risk = tp_risk / len(flagged_risk) if len(flagged_risk) > 0 else 0
    rec_risk = tp_risk / total_fraud if total_fraud > 0 else 0
    
    cost_risk = fp_risk * 50
    loss_avoided_risk = flagged_risk[flagged_risk["isFraud"] == 1]["TransactionAmt"].sum()
    
    results.append({
        "Top %": f"Top {p}%",
        "Threshold": threshold_risk,
        "Flagged": len(flagged_risk),
        "TP": tp_risk,
        "FP": fp_risk,
        "Precision": prec_risk,
        "Recall": rec_risk,
        "Review Cost (₹)": cost_risk,
        "Fraud Avoided (₹)": loss_avoided_risk
    })
    
    # Ring-specific recall calculations (Fused vs Velocity-only)
    threshold_vel = np.percentile(df["velocity_score"], 100 - p)
    
    caught_fused = injected_ring_txns[injected_ring_txns["final_risk_score"] >= threshold_risk]
    caught_vel = injected_ring_txns[injected_ring_txns["velocity_score"] >= threshold_vel]
    
    ring_results.append({
        "Top %": f"Top {p}%",
        "Caught by Velocity": len(caught_vel) / total_injected,
        "Caught by Fused": len(caught_fused) / total_injected,
        "Incremental Graph Lift": (len(caught_fused) - len(caught_vel)) / total_injected
    })

print("="*90)
print("  OPERATIONAL EVALUATION: FUSED RISK SCORE (final_risk_score)")
print("="*90)
print(f"{'Threshold':<10} | {'Flagged':<8} | {'TP':<6} | {'FP':<6} | {'Precision':<9} | {'Recall':<8} | {'Review Cost':<12} | {'Fraud Avoided'}")
print("-" * 90)
for r in results:
    print(f"{r['Top %']:<10} | {r['Flagged']:<8,} | {r['TP']:<6,} | {r['FP']:<6,} | {r['Precision']:<9.1%} | {r['Recall']:<8.1%} | {r['Review Cost (₹)']:<11,.0f} | {r['Fraud Avoided (₹)']:,.0f}")

print("\n" + "="*80)
print("  RING-SPECIFIC RECALL: VELOCITY vs. FUSED (Injected Rings Only)")
print("="*80)
print(f"{'Threshold':<10} | {'Velocity-Only':<15} | {'Fused Score':<15} | {'Graph Incremental Lift'}")
print("-" * 80)
for r in ring_results:
    print(f"{r['Top %']:<10} | {r['Caught by Velocity']:<15.1%} | {r['Caught by Fused']:<15.1%} | {r['Incremental Graph Lift']:+.1%}")
print("="*80 + "\n")
