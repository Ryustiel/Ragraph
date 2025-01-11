import matplotlib.pyplot as plt
import networkx as nx
from ragraph import PairManager

def display(pair: PairManager):

    # Retrieve nodes and edges
    nodes = list(pair.get_nodes())
    edges = list(pair.get_edges())

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes with attributes
    for node in nodes:
        G.add_node(node["id"], type=node["type"], label=node["label"])

    # Add edges with attributes
    for edge in edges:
        G.add_edge(edge["from"], edge["to"], type=edge["type"], label=edge["label"])

    # Define colors for node types
    color_map = []
    for node in G.nodes(data=True):
        if node[1]["type"] == "accessor":
            color_map.append('yellow')
        elif node[1]["type"] == "content":
            color_map.append('purple')
        else:
            color_map.append('grey')  # Fallback color

    # Define edge styles
    solid_edges = []
    solid_colors = []
    dashed_edges = []
    dashed_colors = []
    edge_labels = {}

    for u, v, data in G.edges(data=True):
        if data["type"] == "child":
            solid_edges.append((u, v))
            solid_colors.append('purple')
        elif data["type"] == "neighbor":
            dashed_edges.append((u, v))
            dashed_colors.append('yellow')
            edge_labels[(u, v)] = data["label"]

    # Position nodes using a layout
    pos = nx.spring_layout(G, seed=42)  # You can choose other layouts

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=800)

    # Draw labels
    labels = {node[0]: node[1]["label"] for node in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    # Draw solid edges (child relationships)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=solid_edges,
        edge_color=solid_colors,
        style='solid'
    )

    # Draw dashed edges (neighbor relationships)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=dashed_edges,
        edge_color=dashed_colors,
        style='dashed'
    )

    # Draw edge labels for neighbor edges
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='black',
        font_size=8
    )

    # Customize the plot
    plt.axis('off')
    plt.title("Graph Visualization")
    plt.show()