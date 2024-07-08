import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor
import os


class TicketsGraph:
    def __init__(self, tickets) -> None:
        self.tickets = tickets
        self.preprocessor = TextPreprocessor()
        self.preprocessed_tickets = self.preprocessor.remove_stop_words_from_and_lemmatise_tickets(tickets)
        self.G = self.create_graph()

    def calculate_tfidf_matrix(self):
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.preprocessed_tickets)

    def calculate_cosine_similarity(self, tfidf_matrix):
        return cosine_similarity(tfidf_matrix)

    def create_graph(self):
        tfidf_matrix = self.calculate_tfidf_matrix()
        cosine_sim = self.calculate_cosine_similarity(tfidf_matrix)

        G = nx.Graph()

        for i, ticket in enumerate(self.preprocessed_tickets):
            G.add_node(i, label=self.tickets[i])

        for i in range(len(self.preprocessed_tickets)):
            for j in range(i + 1, len(self.preprocessed_tickets)):
                weight = cosine_sim[i, j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        return G

    def plot_ticket_graph(self, output_path="outputs/ticket_graph.png"):
        pos = nx.spring_layout(self.G)

        plt.figure(figsize=(12, 8))

        node_labels = nx.get_node_attributes(self.G, "label")
        node_labels = {
            node_number: node_label[:20] + "..."
            for node_number, node_label in node_labels.items()
        }
        nx.draw_networkx_nodes(
            self.G, pos, node_size=5000, node_color="skyblue", alpha=0.7
        )
        nx.draw_networkx_labels(
            self.G, pos, labels=node_labels, font_size=10, font_family="sans-serif"
        )

        edges = self.G.edges(data=True)
        for u, v, d in edges:
            weight = d["weight"]
            nx.draw_networkx_edges(
                self.G, pos, edgelist=[(u, v)], width=weight * 10, alpha=0.5
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
