"""
Layer 2: Graph Construction
----------------------------
Turns accounts.csv into a weighted graph using networkx.
- Nodes = accounts
- Edges = shared attributes, weighted by risk level:
    ip, address       -> LOW weight  (0.2) - could be innocent (household)
    device_id, card   -> HIGH weight (0.9) - rarely innocent
"""

import pandas as pd
import networkx as nx
from itertools import combinations

WEIGHTS = {
    "ip": 0.2,
    "address": 0.2,
    "device_id": 0.9,
    "card": 0.9,
}

def build_graph(accounts_df):
    G = nx.Graph()
    G.add_nodes_from(accounts_df["account_id"])

    for attribute, weight in WEIGHTS.items():
        # Group accounts by shared attribute value
        groups = accounts_df.groupby(attribute)["account_id"].apply(list)
        for value, members in groups.items():
            if len(members) < 2:
                continue  # unique value, no sharing, no edge
            # Connect every pair of accounts sharing this value
            for a, b in combinations(members, 2):
                if G.has_edge(a, b):
                    # Already connected via another attribute -> add weight
                    G[a][b]["weight"] += weight
                    G[a][b]["shared_on"] += f",{attribute}"
                else:
                    G.add_edge(a, b, weight=weight, shared_on=attribute)

    return G


def main():
    accounts_df = pd.read_csv("accounts.csv")
    G = build_graph(accounts_df)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Show a few example edges so you can see what's connecting to what
    print("\nSample edges (account pairs and why they're connected):")
    for i, (a, b, data) in enumerate(G.edges(data=True)):
        if i >= 5:
            break
        print(f"  {a} -- {b}  | weight={data['weight']:.1f}  shared_on={data['shared_on']}")


    nx.write_gexf(G, "account_graph.gexf")  # so you can inspect/visualize later
    print("\nSaved graph to account_graph.gexf")


if __name__ == "__main__":
    main()
