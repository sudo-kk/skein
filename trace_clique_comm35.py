import networkx as nx

# We already know comm_to_nodes[35] has 43 nodes.
# Let's recreate the clique recovery for this community exactly as in phase3_ring_detector.py
import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "TransactionDT", "card1", "DeviceType", "DeviceInfo", "is_injected_ring", "ring_id"])
valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)

RARITY_THRESHOLD = 55
TIME_WINDOW = 6 * 3600
MIN_CLUSTER_SIZE = 3
MAX_RING_SIZE = 12
TIGHT_WINDOW = 1800
MIN_CLIQUE_DENSITY = 0.8

combo_counts = valid["combo"].value_counts()
rare_combos = set(combo_counts[combo_counts < RARITY_THRESHOLD].index)
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
            if abs(t2 - t1) <= TIME_WINDOW:
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
print(f"SubG size: {subG.number_of_nodes()}, edges: {subG.number_of_edges()}")

cliques = list(nx.find_cliques(subG))
filtered_cliques = [c for c in cliques if MIN_CLUSTER_SIZE <= len(c) <= MAX_RING_SIZE]
print(f"Maximal cliques in size range [3, 12]: {len(filtered_cliques)}")

card1_to_times = defaultdict(list)
for row in rare_tx.itertuples():
    card1_to_times[row.card1].append(row.TransactionDT)

passed_cliques = []
for clique in filtered_cliques:
    sub = subG.subgraph(clique)
    n = sub.number_of_nodes()
    possible_edges = n * (n - 1) / 2
    density = sub.number_of_edges() / possible_edges
    
    if density >= MIN_CLIQUE_DENSITY:
        times = []
        for node in clique:
            times.extend(card1_to_times[node])
        
        times = sorted(times)
        tight_burst = False
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if times[j] - times[i] <= TIGHT_WINDOW:
                    tight_burst = True
                    break
            if tight_burst:
                break
                
        if tight_burst:
            passed_cliques.append(clique)
        else:
            ring27_members = {np.int64(9921), np.int64(8705), np.int64(4547), np.int64(10631), np.int64(6582), np.int64(5726)}
            if set(clique) & ring27_members:
                print(f"Ring 27 clique rejected due to tight burst window! Clique density: {density:.2f}")

print(f"Cliques passed: {len(passed_cliques)}")
