import pandas as pd
import json

df = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "card1", "DeviceType", "DeviceInfo", "is_injected_ring", "ring_id"])
with open("graph_clusters.json") as f:
    cd = json.load(f)

card1_to_cid = {int(k): v for k, v in cd["card1_to_cid"].items()}
flagged_card1s = set(card1_to_cid.keys())

total_txns = len(df)
unique_card1s = df["card1"].nunique()
txns_with_flagged_card1 = df[df["card1"].isin(flagged_card1s)]
num_txns_flagged_card1 = len(txns_with_flagged_card1)

valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["device_combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)
combo_counts = valid["device_combo"].value_counts()

cluster_combos = set()
for cid, meta in cd["cluster_meta"].items():
    cluster_combos.update(meta["combos"])

df["full_device_combo"] = df["DeviceType"].astype(str) + "|" + df["DeviceInfo"].astype(str)
txns_on_cluster_device_and_card = df[df["card1"].isin(flagged_card1s) & df["full_device_combo"].isin(cluster_combos)]

# Part 2 metrics
print("=== PART 2 ===")
print("Flagged card1 entity count:", len(flagged_card1s))
print("Total unique card1 entity count:", unique_card1s)
print("Entity coverage percentage:", f"{len(flagged_card1s) / unique_card1s * 100:.4f}%")
print("Transaction numerator:", num_txns_flagged_card1)
print("Transaction denominator:", total_txns)
print("Transaction coverage percentage:", f"{num_txns_flagged_card1 / total_txns * 100:.4f}%")
print("Number of transactions that directly used the rare DeviceType+DeviceInfo fingerprint:", len(txns_on_cluster_device_and_card))

louvain_sizes = [meta["size"] for meta in cd["cluster_meta"].values() if meta["source"] == "louvain"]
clique_sizes = [meta["size"] for meta in cd["cluster_meta"].values() if meta["source"] == "clique_recovery"]

print("Size of the largest surviving Louvain/detected cluster:", max(louvain_sizes) if louvain_sizes else 0)
print("Size of the largest recovered clique:", max(clique_sizes) if clique_sizes else 0)

# Check for discarded oversized communities still being treated as detected rings
print("Are there any clusters with size > 12?", any(s > 12 for s in louvain_sizes + clique_sizes))

# Part 3 metrics
print("\n=== PART 3: Ring 27 ===")
ring27 = df[df["ring_id"] == 27]
if ring27.empty:
    print("Ring 27 not found!")
else:
    combos = ring27["full_device_combo"].unique()
    print("DeviceType+DeviceInfo combination(s):", combos)
    
    for combo in combos:
        count = combo_counts.get(combo, 0)
        print(f"Total count of '{combo}' in dataset:", count)
        print("Is count below 55?", count < 55)
        
    print("Do ring members have graph nodes? ", ring27["card1"].isin(flagged_card1s).any())
    # Note: To check if edges exist between members and exactly why it was lost, we already
    # know from the count < 55. If count >= 55, no edges were formed in Phase 3.
