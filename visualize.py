"""
Layer 5 (partial): Visualization
-----------------------------------
Draws the account graph, highlighting which clusters were flagged as
abuse rings vs which are innocent (e.g. households) that correctly
were NOT flagged.

Only connected accounts are shown (isolated accounts with no shared
signals are excluded) -- with 2000 accounts, showing everyone would
just be visual noise. This is a design choice worth being able to
explain: we're visualizing "the interesting part of the graph."
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

WEIGHTS = {"ip": 0.2, "address": 0.2, "device_id": 0.9, "card": 0.9}
RING_WEIGHT_THRESHOLD = 0.5
MIN_COMPONENT_SIZE = 3


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


def flag_components(G):
    flagged_nodes = set()
    for component in nx.connected_components(G):
        if len(component) < MIN_COMPONENT_SIZE:
            continue
        subgraph = G.subgraph(component)
        total_weight = sum(d["weight"] for _, _, d in subgraph.edges(data=True))
        n_edges = subgraph.number_of_edges()
        avg_weight = total_weight / n_edges if n_edges else 0
        if avg_weight >= RING_WEIGHT_THRESHOLD:
            flagged_nodes.update(component)
    return flagged_nodes


def main():
    accounts_df = pd.read_csv("accounts.csv")
    G = build_graph(accounts_df)

    flagged = flag_components(G)

    # Instead of one crowded layout, arrange each connected cluster in its
    # own small area so individual rings/households are actually readable.
    components = [c for c in nx.connected_components(G) if len(c) >= 2]
    components.sort(key=len, reverse=True)

    import math
    n = len(components)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    pos = {}
    cluster_labels = {}
    for i, comp in enumerate(components):
        row, col = divmod(i, cols)
        cx, cy = col * 3, -row * 3  # spacing between cluster "cells"
        subgraph = G.subgraph(comp)
        local_pos = nx.spring_layout(subgraph, seed=42, k=0.6)
        # scale down and offset into this cluster's cell
        for node, (x, y) in local_pos.items():
            pos[node] = (cx + x * 0.8, cy + y * 0.8)
        cluster_labels[i] = (cx, cy + 0.9)

    G_sub = G.subgraph(set().union(*components))

    node_colors = ["red" if n_ in flagged else "steelblue" for n_ in G_sub.nodes()]
    edge_colors = ["red" if G_sub[u][v]["weight"] >= 0.5 else "gray" for u, v in G_sub.edges()]
    edge_widths = [1.5 + G_sub[u][v]["weight"] * 2 for u, v in G_sub.edges()]

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_edges(G_sub, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=180,
                            edgecolors="black", linewidths=0.5, alpha=0.95)

    plt.title(
        f"Abuse-Ring Sentinel — Each Connected Cluster Shown Separately\n"
        f"Red = flagged as abuse ring  |  Blue = connected but NOT flagged (e.g. household)\n"
        f"{len(flagged)} accounts flagged across {sum(1 for c in components if any(x in flagged for x in c))} predicted rings, "
        f"out of {n} total connected clusters",
        fontsize=13
    )
    plt.axis("off")
    plt.axis("equal")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Flagged (predicted ring)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='Not flagged (e.g. household)'),
        Line2D([0], [0], color='red', lw=2, label='High-risk shared signal (device/card)'),
        Line2D([0], [0], color='gray', lw=2, label='Low-risk shared signal (ip/address)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig("ring_visualization.png", dpi=150, bbox_inches="tight")
    print(f"Saved ring_visualization.png")
    print(f"Showing {n} separate clusters ({len(G_sub.nodes())} connected accounts total, out of {len(accounts_df)})")
    print(f"Flagged as rings: {len(flagged)}")


if __name__ == "__main__":
    main()
