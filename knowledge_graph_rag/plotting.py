import networkx as nx
import matplotlib.pyplot as plt
import os


class Plotter:

    def plot_ticket_graph(self, graph, output_path):
        pos = nx.spring_layout(graph)

        plt.figure(figsize=(12, 8))

        node_labels = nx.get_node_attributes(graph, "label")
        node_labels = {
            node_number: f"{node_label[:20]}..."
            for node_number, node_label in node_labels.items()
        }
        nx.draw_networkx_nodes(
            graph, pos, node_size=5000, node_color="skyblue", alpha=0.7
        )
        nx.draw_networkx_labels(
            graph, pos, labels=node_labels, font_size=10, font_family="sans-serif"
        )

        edges = graph.edges(data=True)
        for u, v, d in edges:
            weight = d["weight"]
            nx.draw_networkx_edges(
                graph, pos, edgelist=[(u, v)], width=weight * 10, alpha=0.5
            )
            edge_label = f"{weight:.4f}"
            mid_edge = (pos[u] + pos[v]) / 2
            plt.text(
                mid_edge[0],
                mid_edge[1],
                edge_label,
                fontsize=9,
                ha="center",
                va="center",
            )

        plt.axis("off")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def plot_kn_graph(self, graph, output_path="outputs/kn_graph.png"):
        pos = nx.spring_layout(graph)
        plt.figure(figsize=(12, 8))

        nx.draw_networkx_nodes(
            graph, pos, node_size=5000, node_color="skyblue", alpha=0.7
        )
        node_labels = {
            node: node[:20] + "..." if len(node) > 20 else node
            for node in graph.nodes()
        }
        nx.draw_networkx_labels(
            graph, pos, labels=node_labels, font_size=10, font_family="sans-serif"
        )

        edges = graph.edges(data=True)
        for u, v, d in edges:
            weight = d.get("weight", 1)
            nx.draw_networkx_edges(
                graph, pos, edgelist=[(u, v)], width=weight, alpha=0.5
            )

            edge_label = (
                f"{d['relationship'][:20]}...\n(w:{weight:.2f})"
                if len(d["relationship"]) > 20
                else f"{d['relationship']}\n(w:{weight:.2f})"
            )
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            plt.text(
                x,
                y,
                edge_label,
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        plt.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
