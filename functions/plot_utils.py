from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx


def plot_node_anomaly_graph(
    edge_index,
    pos,
    y,
    title: str = "Graph with node anomalies",
    node_size: int = 90,
):
    """Plot a graph coloring normal nodes green and anomalous nodes red."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(y)))
    graph.add_edges_from(edge_index.T.tolist())

    node_pos = {i: (float(pos[i, 0]), float(pos[i, 1])) for i in range(len(y))}
    node_colors = ["red" if int(label) == 1 else "seagreen" for label in y]

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_edges(graph, node_pos, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(
        graph,
        node_pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="white",
        linewidths=0.8,
    )
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.show()

