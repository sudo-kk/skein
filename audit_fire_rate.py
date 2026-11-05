import pandas as pd
import json

df = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "card1", "DeviceType", "DeviceInfo"])
with open("graph_clusters.json") as f:
    cd = json.load(f)

card1_to_cid = {int(k): v for k, v in cd["card1_to_cid"].items()}
flagged_card1s = set(card1_to_cid.keys())

total_txns = len(df)
unique_card1s = df["card1"].nunique()

txns_with_flagged_card1 = df[df["card1"].isin(flagged_card1s)]
num_txns_flagged_card1 = len(txns_with_flagged_card1)

print("=== AUDIT: TRANSACTION FIRE RATE CALCULATION ===")
print(f"1. Total transactions in dataset (all): {total_txns:,}")
print(f"2. Total unique card1 entities in dataset: {unique_card1s:,}")
print(f"3. Flagged card1 entities in detected clusters: {len(flagged_card1s):,}")
print(f"4. Card1 entity coverage %: {len(flagged_card1s) / unique_card1s * 100:.4f}% ({len(flagged_card1s)} / {unique_card1s})")
print(f"5. Total transactions associated with flagged card1s: {num_txns_flagged_card1:,}")
print(f"6. Reported Fire Rate calculation:")
print(f"   Formula: len(df[df['card1'].isin(flagged_card1s)]) / len(df) * 100")
print(f"   Numerator: {num_txns_flagged_card1:,}")
print(f"   Denominator: {total_txns:,}")
print(f"   Result: {num_txns_flagged_card1 / total_txns * 100:.4f}%\n")

# Check transactions with valid device combo
valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["device_combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)
combo_counts = valid["device_combo"].value_counts()
rare_combos = set(combo_counts[combo_counts < 55].index)

cluster_combos = set()
for cid, meta in cd["cluster_meta"].items():
    cluster_combos.update(meta["combos"])

df["full_device_combo"] = df["DeviceType"].astype(str) + "|" + df["DeviceInfo"].astype(str)
txns_on_cluster_device_and_card = df[df["card1"].isin(flagged_card1s) & df["full_device_combo"].isin(cluster_combos)]

print("=== DEEPER BREAKDOWN ===")
print(f"Total transactions with DeviceType/DeviceInfo present: {len(valid):,} ({len(valid)/total_txns*100:.2f}% of all txns)")
print(f"Transactions where card1 is flagged AND the transaction used the cluster's rare device combo: {len(txns_on_cluster_device_and_card):,}")
print(f"  -> As % of all transactions: {len(txns_on_cluster_device_and_card) / total_txns * 100:.4f}%")
print(f"  -> As % of device-present transactions: {len(txns_on_cluster_device_and_card) / len(valid) * 100:.4f}%")

print("\n=== WHY DID 410 CARD1s PRODUCE 164,521 TRANSACTIONS? ===")
top_cards = txns_with_flagged_card1["card1"].value_counts().head(10)
for c, count in top_cards.items():
    cid = card1_to_cid[c]
    meta = cd["cluster_meta"][str(cid)]
    src = meta["source"]
    sz = meta["size"]
    combos = meta["combos"]
    print(f"card1={c:5d}: {count:6,} txns ({count/total_txns*100:.2f}% of total dataset) | Cluster #{cid:2d} (src={src}, size={sz}, combos={combos[:1]})")
