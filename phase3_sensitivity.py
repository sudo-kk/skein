"""
Phase 3 Fix & Validation
=========================
Part 1: Rarity threshold sensitivity analysis (<50, <=50, <55, <60)
Part 2: Oversized community recovery (clique extraction, recursive Louvain)
Part 3: Combined metrics with best approach
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations
import community as community_louvain
import time as _time

SEED             = 42
MIN_CLUSTER_SIZE = 3
MAX_RING_SIZE    = 12
TIME_WINDOW      = 6 * 3600   # 21600s = 6h
MAX_TEST_THRESH  = 60          # highest rarity threshold we test

print("=" * 70)
print("  Phase 3 Fix & Validation")
print("=" * 70)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("\n[1/5] Loading data ...")
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
print(f"  Ground truth: {total_rings} rings, {len(all_ring_card1s)} ring card1s")

# ================================================================
# 2. PRE-COMPUTE TEMPORAL EDGES (once, for all combos < 60)
# ================================================================
print(f"\n[2/5] Pre-computing temporal edges for all combos < {MAX_TEST_THRESH} ...")
t0 = _time.time()

valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["device_combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)
combo_counts = valid["device_combo"].value_counts()

candidate_combos = set(combo_counts[combo_counts < MAX_TEST_THRESH].index)
candidate_valid = valid[valid["device_combo"].isin(candidate_combos)]


def times_within_window(t1, t2, w):
    for t in t1:
        idx = np.searchsorted(t2, t)
        if idx < len(t2) and abs(int(t2[idx]) - int(t)) <= w: return True
        if idx > 0 and abs(int(t2[idx-1]) - int(t)) <= w: return True
    return False


# Pre-compute: (combo, card1_a, card1_b, combo_count) for all temporal-passing pairs
all_temporal_edges = []
for combo, grp in candidate_valid.groupby("device_combo"):
    cards = grp["card1"].unique()
    if len(cards) < 2:
        continue
    ct = {c: np.sort(grp.loc[grp["card1"] == c, "TransactionDT"].values) for c in cards}
    cl = list(cards)
    cnt = int(combo_counts[combo])
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            if times_within_window(ct[cl[i]], ct[cl[j]], TIME_WINDOW):
                all_temporal_edges.append((combo, cl[i], cl[j], cnt))

print(f"  Total temporal-passing edges (< {MAX_TEST_THRESH}): {len(all_temporal_edges):,}")
print(f"  ({_time.time()-t0:.1f}s)")

# ================================================================
# 3. SENSITIVITY ANALYSIS
# ================================================================
print(f"\n[3/5] Sensitivity analysis ...")


def build_graph(max_count):
    """Build graph from edges where combo_count < max_count."""
    G = nx.Graph()
    for combo, c1, c2, cnt in all_temporal_edges:
        if cnt < max_count:
            if G.has_edge(c1, c2):
                G[c1][c2]["weight"] += 1
                G[c1][c2]["shared_combos"].append(combo)
            else:
                G.add_edge(c1, c2, weight=1, shared_combos=[combo])
    return G


def evaluate_at_threshold(G, label):
    """Run Louvain, split into rings/oversized, compute metrics."""
    if G.number_of_nodes() == 0:
        return {"label": label, "n_clusters": 0, "n_oversized": 0,
                "ring_recall_full": 0, "ring_recall_partial": 0,
                "entity_precision": 0, "fire_rate": 0}

    partition = community_louvain.best_partition(G, random_state=SEED)
    louv = defaultdict(set)
    for node, cid in partition.items():
        louv[cid].add(node)

    rings_det = [c for c in louv.values() if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
    oversized = [c for c in louv.values() if len(c) > MAX_RING_SIZE]

    # Entity precision/recall on ring-sized clusters only
    flagged = set()
    for c in rings_det:
        flagged.update(c)
    tp = flagged & all_ring_card1s
    entity_prec = len(tp) / len(flagged) if flagged else 0
    entity_rec  = len(tp) / len(all_ring_card1s) if all_ring_card1s else 0

    # Ring-level recall
    fully = 0
    partial = 0
    for rid, rset in ring_cards.items():
        best = max((len(rset & c) for c in rings_det), default=0)
        if best == len(rset): fully += 1
        if best >= 2: partial += 1

    # Fire rate
    surviving_cards = flagged
    fire = len(df[df["card1"].isin(surviving_cards)]) / len(df) * 100

    # Oversized stats: how many ring card1s are stuck in oversized communities
    stuck_in_oversized = set()
    for c in oversized:
        stuck_in_oversized.update(c & all_ring_card1s)

    return {
        "label": label,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "n_clusters": len(rings_det),
        "n_oversized": len(oversized),
        "oversized_sizes": sorted([len(c) for c in oversized], reverse=True),
        "ring_recall_full": fully / total_rings,
        "ring_recall_partial": partial / total_rings,
        "entity_precision": entity_prec,
        "entity_recall": entity_rec,
        "fire_rate": fire,
        "rings_in_oversized": len(stuck_in_oversized),
        "partition": partition,
        "louv_map": dict(louv),
        "rings_det": rings_det,
        "oversized": oversized,
    }


# Thresholds: <50 means max_count=50, <=50 means max_count=51, etc.
thresholds = [
    (50,  "<50"),
    (51,  "<=50"),
    (55,  "<55"),
    (60,  "<60"),
]

results = {}
for max_count, label in thresholds:
    t0 = _time.time()
    G = build_graph(max_count)
    r = evaluate_at_threshold(G, label)
    r["graph"] = G
    results[label] = r
    print(f"\n  {label:>5s}: {r['nodes']:4d} nodes, {r['edges']:5d} edges, "
          f"{r['n_clusters']:2d} clusters, {r['n_oversized']:2d} oversized  "
          f"({_time.time()-t0:.1f}s)")

# Summary table
print(f"\n  {'Thresh':>7s} {'Clust':>6s} {'Over':>5s} {'FullRec':>8s} {'PartRec':>8s} "
      f"{'EntPrec':>8s} {'EntRec':>7s} {'Fire%':>6s} {'StuckInOvr':>11s}")
print(f"  {'-'*7} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*11}")
for _, label in thresholds:
    r = results[label]
    ovr_str = str(r["oversized_sizes"][:3]) if r["oversized_sizes"] else "[]"
    print(f"  {label:>7s} {r['n_clusters']:6d} {r['n_oversized']:5d} "
          f"{r['ring_recall_full']:8.1%} {r['ring_recall_partial']:8.1%} "
          f"{r['entity_precision']:8.1%} {r['entity_recall']:7.1%} "
          f"{r['fire_rate']:6.2f} {r['rings_in_oversized']:5d} {ovr_str}")


# ================================================================
# 4. OVERSIZED COMMUNITY RECOVERY
# ================================================================
print(f"\n\n[4/5] Investigating oversized community recovery ...")

# Test recovery at each threshold
for _, label in thresholds:
    r = results[label]
    G = r["graph"]
    oversized = r["oversized"]

    if not oversized:
        print(f"\n  {label}: No oversized communities to process.")
        continue

    print(f"\n  === {label}: {len(oversized)} oversized communities ===")

    for oi, community_set in enumerate(sorted(oversized, key=len, reverse=True)):
        subg = G.subgraph(community_set).copy()
        ring_in_comm = community_set & all_ring_card1s
        ring_ids_in = set(card_to_ring[c] for c in ring_in_comm)

        print(f"\n    Community #{oi+1}: {len(community_set)} nodes, "
              f"{subg.number_of_edges()} edges, "
              f"{len(ring_in_comm)} ring card1s from rings {sorted(ring_ids_in)}")

        # ----- Strategy A: Maximal Clique Extraction -----
        all_cliques = list(nx.find_cliques(subg))
        valid_cliques = [set(c) for c in all_cliques
                         if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
        valid_cliques.sort(key=len, reverse=True)

        # Greedy assignment: largest clique first, no node reuse
        assigned = set()
        final_cliques = []
        for clique in valid_cliques:
            if clique - assigned:  # at least one unassigned node
                clique_remaining = clique - assigned
                if len(clique_remaining) >= MIN_CLUSTER_SIZE:
                    final_cliques.append(clique_remaining)
                    assigned.update(clique_remaining)

        # Also try: keep full cliques, skip if majority already assigned
        assigned2 = set()
        final_cliques2 = []
        for clique in valid_cliques:
            overlap = len(clique & assigned2)
            if overlap <= len(clique) // 2:  # majority unassigned
                final_cliques2.append(clique)
                assigned2.update(clique)

        # Evaluate clique strategy
        clique_flagged = set()
        for c in final_cliques2:
            clique_flagged.update(c)
        clique_tp = clique_flagged & all_ring_card1s
        clique_prec = len(clique_tp) / len(clique_flagged) if clique_flagged else 0
        clique_rings_found = set()
        for rid, rset in ring_cards.items():
            if rid not in ring_ids_in:
                continue
            best = max((len(rset & c) for c in final_cliques2), default=0)
            if best == len(rset):
                clique_rings_found.add(rid)

        print(f"      Clique extraction: {len(valid_cliques)} valid cliques, "
              f"{len(final_cliques2)} after dedup")
        print(f"        Flagged card1s: {len(clique_flagged)}, "
              f"TP: {len(clique_tp)}, Precision: {clique_prec:.2%}")
        print(f"        Rings fully recovered: {sorted(clique_rings_found)} "
              f"({len(clique_rings_found)}/{len(ring_ids_in)})")

        # ----- Strategy B: Recursive Louvain (higher resolution) -----
        for res in [2.0, 5.0, 10.0]:
            sub_part = community_louvain.best_partition(subg, resolution=res,
                                                        random_state=SEED)
            sub_comms = defaultdict(set)
            for node, cid in sub_part.items():
                sub_comms[cid].add(node)
            sub_rings = [c for c in sub_comms.values()
                         if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
            sub_over  = [c for c in sub_comms.values() if len(c) > MAX_RING_SIZE]

            sub_flagged = set()
            for c in sub_rings:
                sub_flagged.update(c)
            sub_tp = sub_flagged & all_ring_card1s
            sub_prec = len(sub_tp) / len(sub_flagged) if sub_flagged else 0

            sub_rings_found = set()
            for rid, rset in ring_cards.items():
                if rid not in ring_ids_in:
                    continue
                best = max((len(rset & c) for c in sub_rings), default=0)
                if best == len(rset):
                    sub_rings_found.add(rid)

            print(f"      Recursive Louvain (res={res}): {len(sub_rings)} subclusters, "
                  f"{len(sub_over)} still oversized")
            print(f"        Flagged: {len(sub_flagged)}, TP: {len(sub_tp)}, "
                  f"Prec: {sub_prec:.2%}")
            print(f"        Rings recovered: {sorted(sub_rings_found)} "
                  f"({len(sub_rings_found)}/{len(ring_ids_in)})")


# ================================================================
# 5. COMBINED EVALUATION: THRESHOLD + BEST RECOVERY
# ================================================================
print(f"\n\n[5/5] Combined evaluation (Louvain + clique recovery) ...")


def evaluate_combined(G, label):
    """Louvain on full graph, then clique extraction on oversized communities."""
    partition = community_louvain.best_partition(G, random_state=SEED)
    louv = defaultdict(set)
    for node, cid in partition.items():
        louv[cid].add(node)

    rings_det = [c for c in louv.values() if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
    oversized = [c for c in louv.values() if len(c) > MAX_RING_SIZE]

    # Recover from oversized via clique extraction
    recovered = []
    for community_set in oversized:
        subg = G.subgraph(community_set)
        cliques = [set(c) for c in nx.find_cliques(subg)
                    if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
        cliques.sort(key=len, reverse=True)
        assigned = set()
        for clique in cliques:
            if len(clique & assigned) <= len(clique) // 2:
                recovered.append(clique)
                assigned.update(clique)

    all_clusters = rings_det + recovered

    # Metrics
    flagged = set()
    for c in all_clusters:
        flagged.update(c)
    tp = flagged & all_ring_card1s
    fp = flagged - all_ring_card1s
    entity_prec = len(tp) / len(flagged) if flagged else 0
    entity_rec  = len(tp) / len(all_ring_card1s)

    fully = 0
    partial = 0
    for rid, rset in ring_cards.items():
        best = max((len(rset & c) for c in all_clusters), default=0)
        if best == len(rset): fully += 1
        if best >= 2: partial += 1

    fire = len(df[df["card1"].isin(flagged)]) / len(df) * 100

    # Cluster-level
    tp_cl = sum(1 for c in all_clusters if c & all_ring_card1s)
    fp_cl = sum(1 for c in all_clusters if not (c & all_ring_card1s))

    return {
        "label": label,
        "louvain_clusters": len(rings_det),
        "recovered_from_oversized": len(recovered),
        "total_clusters": len(all_clusters),
        "oversized_discarded": len(oversized),
        "ring_recall_full": fully / total_rings,
        "ring_recall_partial": partial / total_rings,
        "entity_precision": entity_prec,
        "entity_recall": entity_rec,
        "fire_rate": fire,
        "tp_clusters": tp_cl,
        "fp_clusters": fp_cl,
        "all_clusters": all_clusters,
    }


print(f"\n  {'Thresh':>7s} {'Louv':>5s} {'Recov':>6s} {'Total':>6s} "
      f"{'FullRec':>8s} {'PartRec':>8s} {'EntPrec':>8s} {'EntRec':>7s} "
      f"{'Fire%':>6s} {'TP_cl':>6s} {'FP_cl':>6s}")
print(f"  {'-'*7} {'-'*5} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*7} "
      f"{'-'*6} {'-'*6} {'-'*6}")

combined_results = {}
for max_count, label in thresholds:
    G = results[label]["graph"]
    cr = evaluate_combined(G, label)
    combined_results[label] = cr
    print(f"  {label:>7s} {cr['louvain_clusters']:5d} {cr['recovered_from_oversized']:6d} "
          f"{cr['total_clusters']:6d} "
          f"{cr['ring_recall_full']:8.1%} {cr['ring_recall_partial']:8.1%} "
          f"{cr['entity_precision']:8.1%} {cr['entity_recall']:7.1%} "
          f"{cr['fire_rate']:6.2f} {cr['tp_clusters']:6d} {cr['fp_clusters']:6d}")

# ================================================================
# RECOMMENDATION
# ================================================================
print(f"\n\n{'='*70}")
print(f"  RECOMMENDATION")
print(f"{'='*70}")

# Score each combined result: we want high recall AND reasonable precision
# F1-like score at entity level
for _, label in thresholds:
    cr = combined_results[label]
    p, r = cr["entity_precision"], cr["entity_recall"]
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    print(f"  {label:>7s}: Entity F1={f1:.3f}  (P={p:.3f}, R={r:.3f})  "
          f"Fire={cr['fire_rate']:.2f}%  Clusters={cr['total_clusters']}")

print(f"\n{'='*70}")
print("  Done.")
print(f"{'='*70}")
