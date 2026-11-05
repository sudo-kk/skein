"""
Phase 3 (Final) -- Ring Detector
=================================
Config:
  Rarity threshold:  < 55  (from sensitivity analysis)
  Temporal window:   6h    (for graph edge creation)
  Cluster size:      3-12  (Louvain clusters)
  Clique recovery:   density >= 0.80 AND all members
                     transact within 30 minutes of each other
                     on the same rare device combo.
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import community as community_louvain
import json
import time as _time

SEED             = 42
RARITY_THRESHOLD = 55
MIN_CLUSTER_SIZE = 3
MAX_RING_SIZE    = 12
TIME_WINDOW      = 6 * 3600        # 6h for graph edges
TIGHT_WINDOW     = 30 * 60         # 30 min for clique recovery
MIN_CLIQUE_DENSITY = 0.80

print("=" * 70)
print("  Phase 3 (Final): Ring Detector")
print("=" * 70)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n[1/6] Loading data ...")
t0 = _time.time()
LOAD_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "DeviceType", "DeviceInfo",
    "ring_id", "is_injected_ring", "isFraud",
]
df = pd.read_csv("augmented_with_rings.csv", usecols=LOAD_COLS)
print(f"  {len(df):,} rows  ({_time.time()-t0:.1f}s)")

# Ground truth
ring_df = df[df["is_injected_ring"] == True]
ring_cards = {}
for rid, grp in ring_df.groupby("ring_id"):
    ring_cards[int(rid)] = set(grp["card1"].unique())
card_to_ring = {}
for rid, cards in ring_cards.items():
    for c in cards:
        card_to_ring[c] = rid
all_ring_card1s = set(card_to_ring.keys())
total_rings = len(ring_cards)

# ================================================================
# 2. RARE DEVICE COMBOS
# ================================================================
print(f"\n[2/6] Identifying rare device combos (< {RARITY_THRESHOLD}) ...")
t0 = _time.time()
valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["device_combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)
combo_counts = valid["device_combo"].value_counts()
rare_combos = set(combo_counts[combo_counts < RARITY_THRESHOLD].index)
rare_valid = valid[valid["device_combo"].isin(rare_combos)]
print(f"  Rare combos: {len(rare_combos):,}  ({_time.time()-t0:.1f}s)")

# ================================================================
# 3. BUILD GRAPH (temporal constraint)
# ================================================================
print(f"\n[3/6] Building entity graph (6h temporal window) ...")
t0 = _time.time()

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
                    G[cl[i]][cl[j]]["shared_combos"].append(combo)
                else:
                    G.add_edge(cl[i], cl[j], weight=1, shared_combos=[combo])

print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  "
      f"({_time.time()-t0:.1f}s)")

# ================================================================
# 4. LOUVAIN + CLIQUE RECOVERY WITH TIGHT TEMPORAL CHECK
# ================================================================
print(f"\n[4/6] Louvain + density+temporal-filtered clique recovery ...")
t0 = _time.time()

partition = community_louvain.best_partition(G, random_state=SEED)
louv = defaultdict(set)
for node, cid in partition.items():
    louv[cid].add(node)

louvain_rings = [c for c in louv.values() if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
oversized     = [c for c in louv.values() if len(c) > MAX_RING_SIZE]

print(f"  Louvain ring-sized clusters: {len(louvain_rings)}")
print(f"  Oversized communities:       {len(oversized)}")


def clique_has_tight_temporal_cluster(clique_card1s, graph, rare_valid_df, window_sec):
    """Check if ALL card1s in the clique have transactions within
    `window_sec` of each other on the same rare device combo.

    Uses a sweep-line: merge all (time, card1) events from the
    rare-combo transactions for these card1s, sort by time,
    then slide a window checking if all card1s are represented.
    """
    card1_set = set(clique_card1s)
    target = len(card1_set)

    # Get the rare combos used by edges in this clique
    subg = graph.subgraph(card1_set)
    clique_combos = set()
    for _, _, d in subg.edges(data=True):
        clique_combos.update(d.get("shared_combos", []))

    # Collect (time, card1) events for these combos + card1s
    mask = (rare_valid_df["card1"].isin(card1_set) &
            rare_valid_df["device_combo"].isin(clique_combos))
    subset = rare_valid_df.loc[mask, ["TransactionDT", "card1"]]

    if subset.empty:
        return False

    events = list(zip(subset["TransactionDT"].values, subset["card1"].values))
    events.sort(key=lambda x: x[0])

    # Sweep-line: find any window of `window_sec` containing all card1s
    card_count = defaultdict(int)
    distinct = 0
    left = 0

    for right in range(len(events)):
        t_r, c_r = events[right]
        if card_count[c_r] == 0:
            distinct += 1
        card_count[c_r] += 1

        # Shrink from left
        while int(events[right][0]) - int(events[left][0]) > window_sec:
            t_l, c_l = events[left]
            card_count[c_l] -= 1
            if card_count[c_l] == 0:
                distinct -= 1
            left += 1

        if distinct == target:
            return True

    return False


recovered_cliques = []
cliques_checked = 0
cliques_density_pass = 0
cliques_temporal_pass = 0

for community_set in oversized:
    subg = G.subgraph(community_set)
    raw_cliques = [set(c) for c in nx.find_cliques(subg)
                   if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
    raw_cliques.sort(key=len, reverse=True)

    assigned = set()
    for clique in raw_cliques:
        if len(clique & assigned) <= len(clique) // 2:
            remaining = clique - assigned
            if len(remaining) < MIN_CLUSTER_SIZE:
                continue

            cliques_checked += 1

            # Density check
            sub_sub = G.subgraph(remaining)
            n = len(remaining)
            max_edges = n * (n - 1) / 2
            density = sub_sub.number_of_edges() / max_edges if max_edges > 0 else 0
            if density < MIN_CLIQUE_DENSITY:
                continue
            cliques_density_pass += 1

            # Tight temporal clustering check
            if not clique_has_tight_temporal_cluster(
                    remaining, G, rare_valid, TIGHT_WINDOW):
                continue
            cliques_temporal_pass += 1

            recovered_cliques.append(remaining)
            assigned.update(remaining)

print(f"  Cliques checked:         {cliques_checked}")
print(f"  Passed density >= {MIN_CLIQUE_DENSITY}:  {cliques_density_pass}")
print(f"  Passed temporal <= {TIGHT_WINDOW//60}m: {cliques_temporal_pass}")
print(f"  Recovered cliques:       {len(recovered_cliques)}")
if recovered_cliques:
    rec_sizes = sorted([len(c) for c in recovered_cliques])
    print(f"  Recovered sizes:         {rec_sizes}")

all_clusters = louvain_rings + recovered_cliques
print(f"  TOTAL clusters:          {len(all_clusters)}")
print(f"  ({_time.time()-t0:.1f}s)")

# ================================================================
# 5. BUILD CLUSTER METADATA + MAPPINGS
# ================================================================
print(f"\n[5/6] Building cluster metadata ...")

cluster_meta = {}
card1_to_cid = {}

for i, cluster in enumerate(all_clusters):
    subg = G.subgraph(cluster)
    combos = set()
    for _, _, d in subg.edges(data=True):
        combos.update(d.get("shared_combos", []))
    n = len(cluster)
    max_e = n * (n-1) / 2
    density = subg.number_of_edges() / max_e if max_e > 0 else 0

    cluster_meta[i] = {
        "size": n,
        "combos": sorted(combos)[:3],
        "density": round(density, 3),
        "source": "louvain" if i < len(louvain_rings) else "clique_recovery",
    }
    for card in cluster:
        card1_to_cid[card] = i

# Save for Phase 4
cluster_data = {
    "cluster_meta": {str(k): v for k, v in cluster_meta.items()},
    "card1_to_cid": {str(k): v for k, v in card1_to_cid.items()},
    "config": {
        "RARITY_THRESHOLD": RARITY_THRESHOLD,
        "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,
        "MAX_RING_SIZE": MAX_RING_SIZE,
        "TIME_WINDOW": TIME_WINDOW,
        "TIGHT_WINDOW": TIGHT_WINDOW,
        "MIN_CLIQUE_DENSITY": MIN_CLIQUE_DENSITY,
    }
}
with open("graph_clusters.json", "w") as f:
    json.dump(cluster_data, f)
print(f"  Saved graph_clusters.json  ({len(card1_to_cid)} card1 mappings)")

# ================================================================
# 6. EVALUATION
# ================================================================
print(f"\n[6/6] Evaluation ...")

flagged = set(card1_to_cid.keys())
tp = flagged & all_ring_card1s
fp = flagged - all_ring_card1s
entity_prec = len(tp) / len(flagged) if flagged else 0
entity_rec  = len(tp) / len(all_ring_card1s)

fully = 0
partial = 0
ring_status = {}
for rid, rset in ring_cards.items():
    best_overlap = 0
    best_cluster_size = 0
    best_cid = -1
    for i, cluster in enumerate(all_clusters):
        overlap = len(rset & cluster)
        if overlap > best_overlap:
            best_overlap = overlap
            best_cluster_size = len(cluster)
            best_cid = i
    is_full = best_overlap == len(rset)
    is_partial = best_overlap >= 2
    if is_full: fully += 1
    if is_partial: partial += 1
    ring_status[rid] = {
        "size": len(rset), "overlap": best_overlap,
        "cluster_size": best_cluster_size, "cluster_id": best_cid,
        "status": "FULL" if is_full else ("PARTIAL" if is_partial else "MISS"),
        "source": cluster_meta[best_cid]["source"] if best_cid >= 0 else "none",
    }

fire = len(df[df["card1"].isin(flagged)]) / len(df) * 100

tp_cl = sum(1 for c in all_clusters if c & all_ring_card1s)
fp_cl = sum(1 for c in all_clusters if not (c & all_ring_card1s))

print(f"\n{'='*70}")
print(f"  FINAL PHASE 3 METRICS")
print(f"{'='*70}")
print(f"  Config:")
print(f"    Rarity threshold:    < {RARITY_THRESHOLD}")
print(f"    Temporal window:     {TIME_WINDOW//3600}h (edges)")
print(f"    Tight window:        {TIGHT_WINDOW//60}m (clique recovery)")
print(f"    Cluster size range:  {MIN_CLUSTER_SIZE}-{MAX_RING_SIZE}")
print(f"    Clique min density:  {MIN_CLIQUE_DENSITY}")

print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Clusters: {len(louvain_rings)} Louvain + "
      f"{len(recovered_cliques)} recovered = {len(all_clusters)} total")

print(f"\n  Entity-level metrics:")
print(f"    Flagged card1s:    {len(flagged)}")
print(f"    True positives:    {len(tp)}")
print(f"    False positives:   {len(fp)}")
print(f"    PRECISION:         {entity_prec:.4f}  ({len(tp)}/{len(flagged)})")
print(f"    RECALL:            {entity_rec:.4f}  ({len(tp)}/{len(all_ring_card1s)})")

print(f"\n  Ring-level recall:")
print(f"    Fully recovered:   {fully:3d} / {total_rings}  ({100*fully/total_rings:.1f}%)")
print(f"    Partially (>=2):   {partial:3d} / {total_rings}  ({100*partial/total_rings:.1f}%)")
missed = total_rings - partial
print(f"    Missed:            {missed:3d} / {total_rings}")

print(f"\n  Cluster-level:")
print(f"    TP clusters:       {tp_cl}")
print(f"    FP clusters:       {fp_cl}")
cl_prec = tp_cl / (tp_cl + fp_cl) if (tp_cl + fp_cl) > 0 else 0
print(f"    Cluster precision: {cl_prec:.4f}")

print(f"\n  Fire rate: {fire:.2f}%  ({len(df[df['card1'].isin(flagged)]):,} / {len(df):,})")
print(f"  Card1 flag rate: {len(flagged)}/{df['card1'].nunique()} "
      f"= {100*len(flagged)/df['card1'].nunique():.2f}%")

# Per-ring detail
print(f"\n  Per-ring detail:")
print(f"    {'Ring':>5s} {'Size':>5s} {'Over':>5s} {'ClSz':>5s} {'Source':<15s} {'Status':<8s}")
for rid in sorted(ring_status.keys()):
    rs = ring_status[rid]
    print(f"    {rid:5d} {rs['size']:5d} {rs['overlap']:5d} {rs['cluster_size']:5d} "
          f"{rs['source']:<15s} {rs['status']:<8s}")

# Comparison
print(f"\n  COMPARISON:")
print(f"    {'Metric':<25s} {'Louv-only (<50)':>16s} {'Final':>16s}")
print(f"    {'-'*25} {'-'*16} {'-'*16}")
print(f"    {'Ring recall (full)':<25s} {'47.5%':>16s} {f'{100*fully/total_rings:.1f}%':>16s}")
print(f"    {'Entity precision':<25s} {'50.0%':>16s} {f'{entity_prec:.1%}':>16s}")
print(f"    {'Fire rate (txn %)':<25s} {'3.12%':>16s} {f'{fire:.2f}%':>16s}")
card1_flag_pct = f"{100*len(flagged)/df['card1'].nunique():.2f}%"
print(f"    {'Card1 flag rate':<25s} {'1.4%':>16s} {card1_flag_pct:>16s}")
print(f"    {'Total clusters':<25s} {'39':>16s} {f'{len(all_clusters)}':>16s}")

print(f"\n{'='*70}")
print("  Done.")
print(f"{'='*70}")
