"""
Diagnostic: Why did some rings lose full recovery after the temporal patch?

For each ring that was previously fully recovered but is now missed,
check whether:
  1. All ring members share a DeviceType+DeviceInfo combo
  2. That combo is rare (< 50 transactions)
  3. All pairwise TransactionDT diffs are within 6h (21600s)
  4. Edges actually exist in the patched graph
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import community as community_louvain

SEED             = 42
RARITY_THRESHOLD = 50
MIN_CLUSTER_SIZE = 3
MAX_RING_SIZE    = 12
TIME_WINDOW      = 6 * 3600  # 21600 seconds

# ================================================================
# 1. Load data
# ================================================================
print("Loading data ...")
LOAD_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "DeviceType", "DeviceInfo",
    "ring_id", "is_injected_ring", "isFraud",
]
df = pd.read_csv("augmented_with_rings.csv", usecols=LOAD_COLS)

# ================================================================
# 2. Examine ALL 40 rings: timestamps, device combos, rarity
# ================================================================
ring_df = df[df["is_injected_ring"] == True].copy()
ring_df["device_combo"] = (
    ring_df["DeviceType"].astype(str) + "|" + ring_df["DeviceInfo"].astype(str)
)

# Global combo counts (entire dataset, not just rings)
valid_all = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid_all["device_combo"] = (
    valid_all["DeviceType"].astype(str) + "|" + valid_all["DeviceInfo"].astype(str)
)
combo_counts = valid_all["device_combo"].value_counts()

print(f"\n{'='*80}")
print(f"  RING-BY-RING DIAGNOSTIC")
print(f"  TIME_WINDOW = {TIME_WINDOW}s = {TIME_WINDOW/3600}h")
print(f"  RARITY_THRESHOLD = {RARITY_THRESHOLD}")
print(f"{'='*80}\n")

for ring_id in sorted(ring_df["ring_id"].unique()):
    members = ring_df[ring_df["ring_id"] == ring_id]
    card1s  = members["card1"].unique()
    times   = members["TransactionDT"].values
    combos  = members["device_combo"].unique()
    dev_types = members["DeviceType"].unique()
    dev_infos = members["DeviceInfo"].unique()

    # Time spread within ring
    t_min, t_max = times.min(), times.max()
    t_spread = t_max - t_min

    # Check for NaN device info
    has_nan_device = members["DeviceType"].isna().any() or members["DeviceInfo"].isna().any()

    # Check combo rarity
    combo_str = combos[0] if len(combos) == 1 else str(combos)
    combo_count = combo_counts.get(combos[0], 0) if len(combos) == 1 else -1
    is_rare = combo_count < RARITY_THRESHOLD

    # Check all pairwise time diffs
    max_pairwise_diff = 0
    all_within_window = True
    for t1, t2 in combinations(times, 2):
        diff = abs(int(t1) - int(t2))
        max_pairwise_diff = max(max_pairwise_diff, diff)
        if diff > TIME_WINDOW:
            all_within_window = False

    # Status
    issues = []
    if has_nan_device:
        issues.append("NaN device info")
    if not is_rare:
        issues.append(f"combo NOT rare ({combo_count} >= {RARITY_THRESHOLD})")
    if not all_within_window:
        issues.append(f"time spread {t_spread}s > {TIME_WINDOW}s")

    status = "OK" if not issues else "ISSUE: " + " | ".join(issues)

    print(f"  Ring {ring_id:2d}:  {len(card1s)} members  |  "
          f"time_spread={t_spread:6d}s ({t_spread/3600:.2f}h)  |  "
          f"combo_count={combo_count:3d}  |  rare={is_rare}  |  "
          f"nan_device={has_nan_device}  |  {status}")

    if issues:
        print(f"           combo: {combo_str}")
        print(f"           DeviceType: {dev_types}")
        print(f"           DeviceInfo: {dev_infos}")
        print(f"           TransactionDTs: {sorted(times)}")
        print(f"           card1s: {sorted(card1s)}")
        print()

# ================================================================
# 3. Rebuild the patched graph and check edge presence per ring
# ================================================================
print(f"\n{'='*80}")
print(f"  EDGE-PRESENCE CHECK IN PATCHED GRAPH")
print(f"{'='*80}\n")

rare_combos = set(combo_counts[combo_counts < RARITY_THRESHOLD].index)
rare_valid  = valid_all[valid_all["device_combo"].isin(rare_combos)]

def times_within_window(t1, t2, w):
    for t in t1:
        idx = np.searchsorted(t2, t)
        if idx < len(t2) and abs(int(t2[idx]) - int(t)) <= w: return True
        if idx > 0 and abs(int(t2[idx-1]) - int(t)) <= w: return True
    return False

G = nx.Graph()
for combo, grp in rare_valid.groupby("device_combo"):
    cards = grp["card1"].unique()
    if len(cards) < 2: continue
    ct = {c: np.sort(grp.loc[grp["card1"]==c, "TransactionDT"].values) for c in cards}
    cl = list(cards)
    for i in range(len(cl)):
        for j in range(i+1, len(cl)):
            if times_within_window(ct[cl[i]], ct[cl[j]], TIME_WINDOW):
                if G.has_edge(cl[i], cl[j]):
                    G[cl[i]][cl[j]]["weight"] += 1
                else:
                    G.add_edge(cl[i], cl[j], weight=1)

print(f"  Patched graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# Louvain
partition = community_louvain.best_partition(G, random_state=SEED)
louv_map = defaultdict(set)
for node, cid in partition.items():
    louv_map[cid].add(node)
surviving = {cid: s for cid, s in louv_map.items()
             if MIN_CLUSTER_SIZE <= len(s) <= MAX_RING_SIZE}
card1_to_cid = {}
for cid, cards in surviving.items():
    for c in cards:
        card1_to_cid[c] = cid

for ring_id in sorted(ring_df["ring_id"].unique()):
    members = ring_df[ring_df["ring_id"] == ring_id]
    card1s  = sorted(members["card1"].unique())

    # Check which card1s are in the graph at all
    in_graph = [c for c in card1s if G.has_node(c)]
    not_in_graph = [c for c in card1s if not G.has_node(c)]

    # Check pairwise edges
    n_pairs = 0
    n_edges = 0
    missing_edges = []
    for c1, c2 in combinations(card1s, 2):
        n_pairs += 1
        if G.has_edge(c1, c2):
            n_edges += 1
        else:
            missing_edges.append((c1, c2))

    # Check Louvain cluster assignment
    cluster_ids = set()
    for c in card1s:
        if c in card1_to_cid:
            cluster_ids.add(card1_to_cid[c])

    in_surviving = [c for c in card1s if c in card1_to_cid]
    n_in_surv = len(in_surviving)

    fully_recovered = (n_in_surv == len(card1s) and len(cluster_ids) == 1)

    status = "FULL" if fully_recovered else "MISS"
    if n_in_surv > 0 and not fully_recovered:
        status = "PARTIAL"

    print(f"  Ring {ring_id:2d} ({len(card1s)} members):  "
          f"in_graph={len(in_graph)}/{len(card1s)}  "
          f"edges={n_edges}/{n_pairs}  "
          f"in_surviving_cluster={n_in_surv}/{len(card1s)}  "
          f"louvain_clusters={cluster_ids or 'none'}  "
          f"=> {status}")

    if status != "FULL":
        if not_in_graph:
            print(f"           card1s NOT in graph: {not_in_graph}")
        if missing_edges:
            print(f"           missing edges: {missing_edges[:5]}")
        if len(cluster_ids) > 1:
            print(f"           SPLIT across Louvain clusters: {cluster_ids}")
        # Check why card1s are missing from graph
        for c in not_in_graph:
            c_txns = valid_all[valid_all["card1"] == c]
            if len(c_txns) == 0:
                print(f"           card1={c}: NO device info at all")
            else:
                c_combos = c_txns["device_combo"].unique()
                for co in c_combos:
                    cnt = combo_counts.get(co, 0)
                    print(f"           card1={c}: combo='{co}' count={cnt} "
                          f"rare={cnt < RARITY_THRESHOLD}")
        print()

print(f"\n{'='*80}")
print("  Done.")
print(f"{'='*80}")
