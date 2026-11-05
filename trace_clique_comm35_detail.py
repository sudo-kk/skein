import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "TransactionDT", "card1", "DeviceType", "DeviceInfo", "is_injected_ring", "ring_id"])
valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)

combo_counts = valid["combo"].value_counts()
rare_combos = set(combo_counts[combo_counts < 55].index)
rare_tx = valid[valid["combo"].isin(rare_combos)].copy()
rare_tx.sort_values("TransactionDT", inplace=True)

combo_to_cards = defaultdict(list)
for row in rare_tx.itertuples():
    combo_to_cards[row.combo].append((row.card1, row.TransactionDT))

edges = defaultdict(int)
for combo, events in combo_to_cards.items():
    for i in range(len(events)):
        c1, t1 = events[i]
        for j in range(i + 1, len(events)):
            c2, t2 = events[j]
            if abs(t2 - t1) <= 6 * 3600:
                if c1 != c2:
                    edges[frozenset([c1, c2])] += 1
            else:
                break

G = nx.Graph()
for edge_set, weight in edges.items():
    u, v = list(edge_set)
    G.add_edge(u, v, weight=weight)

import community as community_louvain
partition = community_louvain.best_partition(G, random_state=42)

comm_to_nodes = defaultdict(list)
for node, comm_id in partition.items():
    comm_to_nodes[comm_id].append(node)

comm35_nodes = comm_to_nodes[35]
subG = G.subgraph(comm35_nodes)
cliques = list(nx.find_cliques(subG))

ring27_members = {np.int64(9921), np.int64(8705), np.int64(4547), np.int64(10631), np.int64(6582), np.int64(5726)}

print("Number of cliques found:", len(cliques))
for idx, clique in enumerate(cliques):
    overlap = set(clique) & ring27_members
    if overlap:
        print(f"Clique {idx}: size={len(clique)}, overlap with ring27={len(overlap)}")
        if len(clique) > 12:
            print("  -> Rejected because size > 12")
