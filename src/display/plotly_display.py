"""
A library to display interactive plots of the graphs. Useful as a dev or vizualization tool.
"""

import plotly.graph_objects as go
import networkx as nx

from typing import List
from components.ragraph import PairGraph, NodePlot, EdgePlot


def display_plotly(pair: PairGraph):
    # Retrieve nodes and edges
    nodes: List[NodePlot] = list(pair.get_nodes())
    edges: List[EdgePlot] = list(pair.get_edges())

    G = nx.Graph()

    for node in nodes:
        G.add_node(node.id, label=node.label, chunk=node.chunk, type=node.type)

    for edge in edges:
        G.add_edge(edge.start, edge.end, type=edge.type, label=edge.label)

    pos = nx.spring_layout(G, seed=42)

    # Prepare edge coordinates and styling
    edge_x = []
    edge_y = []
    edge_colors = []  # Will hold the colors of the edges based on type
    edge_dash_styles = []  # Will hold the dash style for each edge
    edge_text = []  # Will hold the text labels for neighbor edges

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        # Process edge based on its type
        edge_type = edge[2].get("type")
        if edge_type == "neighbor":
            edge_colors.append('orange')  # Now set neighbor edges to orange
            edge_dash_styles.append('dash')  # Dashed style
            edge_text.append(edge[2].get("label", ""))  # Store the label for the neighbor edge
        else:
            edge_colors.append('#888')  # Default gray for non-neighbor edges
            edge_dash_styles.append('solid')  # Solid line
            edge_text.append('')  # No label for non-neighbor edges

    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(0, 0, 0, 0)'),  # Initially invisible lines (we'll handle specific colors below)
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        txt = G.nodes[node].get("chunk", "NO CHUNK")
        text.append(txt)

    # Create the node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[G.nodes[node]["label"] for node in G.nodes()],
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            color=['orange' if G.nodes[n]["type"] == "accessor" else 'purple' if G.nodes[n]["type"] == "content" else 'grey' for n in G.nodes()],
            size=20,
            line_width=2)
    )

    # First, add the edge traces in the Figure.
    fig = go.Figure()

    # Plot each edge individually to assign different colors/dash styles/labels
    for i, (x0, y0, x1, y1, color, dash, label) in enumerate(zip(edge_x[0::3], edge_y[0::3], edge_x[1::3], edge_y[1::3], edge_colors, edge_dash_styles, edge_text)):
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=1, color=color, dash=dash),
            hoverinfo='none'
        ))
        if label:  # Add the label text for 'neighbor' edges
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='text',
                text=label,
                textposition="top right",
                hoverinfo='none'
            ))

    # Now add the node trace (so nodes appear on top of edges)
    fig.add_trace(node_trace)

    # Update layout settings
    fig.update_layout(
        title=pair.name,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Add hover text for nodes
    fig.update_traces(
        selector=dict(mode='markers+text'),
        hovertext=text
    )

    fig.show()