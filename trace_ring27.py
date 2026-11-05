import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations

df = pd.read_csv("augmented_with_rings.csv", usecols=["TransactionID", "TransactionDT", "card1", "DeviceType", "DeviceInfo", "is_injected_ring", "ring_id"])

# Recreate graph logic
valid = df.dropna(subset=["DeviceType", "DeviceInfo"]).copy()
valid["combo"] = valid["DeviceType"].astype(str) + "|" + valid["DeviceInfo"].astype(str)

RARITY_THRESHOLD = 55
TIME_WINDOW = 6 * 3600

combo_counts = valid["combo"].value_counts()
rare_combos = set(combo_counts[combo_counts < RARITY_THRESHOLD].index)

rare_tx = valid[valid["combo"].isin(rare_combos)].copy()
rare_tx.sort_values("TransactionDT", inplace=True)

ring27 = df[df["ring_id"] == 27]
ring27_card1s = set(ring27["card1"].unique())
print("Ring 27 members:", ring27_card1s)

combo_to_cards = defaultdict(list)
for row in rare_tx.itertuples():
    combo_to_cards[row.combo].append((row.card1, row.TransactionDT))

edges = defaultdict(int)
for combo, events in combo_to_cards.items():
    if combo == "desktop|MDDRJS":
        print(f"Total events for combo {combo}: {len(events)}")
        r27_events = [e for e in events if e[0] in ring27_card1s]
        print(f"Events for Ring 27 members: {len(r27_events)}")
        
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

print(f"Does G have Ring 27 members? {any(c in G for c in ring27_card1s)}")
if any(c in G for c in ring27_card1s):
    in_g = [c for c in ring27_card1s if c in G]
    print("Members in G:", in_g)
    for c in in_g:
        print(f"Degree of {c}: {G.degree(c)}")
else:
    print("No edges were formed for Ring 27 members.")

import community as community_louvain
partition = community_louvain.best_partition(G, random_state=42)

comm_to_nodes = defaultdict(list)
for node, comm_id in partition.items():
    comm_to_nodes[comm_id].append(node)

for c in in_g:
    comm_id = partition[c]
    comm = comm_to_nodes[comm_id]
    print(f"Node {c} is in community {comm_id} of size {len(comm)}")

