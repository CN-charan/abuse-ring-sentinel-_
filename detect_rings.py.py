"""
Layer 3: Detection
--------------------
Finds connected clusters in the graph and decides which ones look
like abuse rings vs innocent overlaps (like households).

Approach: find connected components (islands), then flag a component
as a ring only if its TOTAL EDGE WEIGHT is high enough -- not just its
size. This is what protects a family-of-5 household (all low-weight
0.2 edges) from being flagged the same as a ring (high-weight 0.9 edges).
"""

import pandas as pd
import networkx as nx
from itertools import combinations

WEIGHTS = {"ip": 0.2, "address": 0.2, "device_id": 0.9, "card": 0.9}

# Tunable: a component is flagged as a ring if its average edge weight
# (total weight / number of edges) exceeds this. High-risk-only edges
# average 0.9; low-risk-only edges average 0.2. This threshold sits
# between them, so mixed/high-risk clusters get flagged, low-risk ones don't.
RING_WEIGHT_THRESHOLD = 0.5
MIN_COMPONENT_SIZE = 3  # ignore isolated pairs, need at least a small group


def build_graph(accounts_df):
    G = nx.Graph()
    G.add_nodes_from(accounts_df["account_id"])
    for attribute, weight in WEIGHTS.items():
        groups = accounts_df.groupby(attribute)["account_id"].apply(list)
        for value, members in groups.items():
            if len(members) < 2:
                continue
            for a, b in combinations(members, 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += weight
                else:
                    G.add_edge(a, b, weight=weight)
    return G


def detect_rings(G):
    """Return a dict: account_id -> predicted_ring_label (or None)."""
    predictions = {node: None for node in G.nodes()}
    flagged_component_count = 0

    for component in nx.connected_components(G):
        if len(component) < MIN_COMPONENT_SIZE:
            continue  # too small to matter, e.g. an isolated pair

        subgraph = G.subgraph(component)
        total_weight = sum(data["weight"] for _, _, data in subgraph.edges(data=True))
        n_edges = subgraph.number_of_edges()
        avg_weight = total_weight / n_edges if n_edges else 0

        if avg_weight >= RING_WEIGHT_THRESHOLD:
            flagged_component_count += 1
            label = f"predicted_ring_{flagged_component_count}"
            for node in component:
                predictions[node] = label

    return predictions


def main():
    accounts_df = pd.read_csv("accounts.csv")
    ground_truth = pd.read_csv("ground_truth.csv")

    G = build_graph(accounts_df)
    predictions = detect_rings(G)

    accounts_df["predicted_ring"] = accounts_df["account_id"].map(predictions)
    accounts_df["predicted_flag"] = accounts_df["predicted_ring"].notna().astype(int)

    n_flagged = accounts_df["predicted_flag"].sum()
    n_flagged_components = accounts_df["predicted_ring"].nunique()

    print(f"Flagged {n_flagged} accounts across {n_flagged_components} predicted rings")

    # Quick peek: how many flagged accounts were ACTUALLY in a real ring?
    merged = accounts_df.merge(ground_truth, on="account_id")
    true_positives = ((merged.predicted_flag == 1) & (merged.is_fraud_ring == 1)).sum()
    false_positives = ((merged.predicted_flag == 1) & (merged.is_fraud_ring == 0)).sum()
    false_negatives = ((merged.predicted_flag == 0) & (merged.is_fraud_ring == 1)).sum()

    print(f"  True positives (correctly flagged ring members): {true_positives}")
    print(f"  False positives (innocent accounts wrongly flagged): {false_positives}")
    print(f"  False negatives (ring members missed): {false_negatives}")

    accounts_df.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv (full evaluation metrics come in Layer 4)")


if __name__ == "__main__":
    main()
