import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor


class TicketsGraph:
    def __init__(self, tickets) -> None:
        self.tickets = tickets
        self.preprocessor = TextPreprocessor()
        self.preprocessed_tickets = self.preprocessor.remove_stop_words_from_and_lemmatise_tickets(tickets)
        self.G = self.create_graph()

    def calculate_tfidf_matrix(self):
        vectorizer = TfidfVectorizer()
        c = self.preprocessed_tickets
        b = vectorizer.fit_transform(self.preprocessed_tickets)
        return vectorizer.fit_transform(self.preprocessed_tickets)

    def calculate_cosine_similarity(self, tfidf_matrix):
        a = cosine_similarity(tfidf_matrix)
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

    def find_connected_documents(self, input_sentence, N=3):
        #TODO: investigate the input_sentence. Currently, the input_sentence shoule be exactly like one of the tickets in the settings.TICKETS list.
        # input_sentence = self.preprocessor.preprocess_text(input_sentence)

        node_index = None
        for node, data in self.G.nodes(data=True):
            if data["label"] == input_sentence:
                node_index = node
                break

        if node_index is None:
            raise ValueError("The provided sentence is not in the graph.")

        neighbors = [
            (neighbor, self.G[node_index][neighbor]["weight"])
            for neighbor in self.G.neighbors(node_index)
        ]

        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
        print(f"Sorted neighbors: {neighbors}")

        top_neighbors = [
            {"ticket": self.G.nodes[neighbor]["label"]}
            for neighbor, weight in neighbors[:N]
        ]
        print(f"Top neighbors: {top_neighbors}")

        return top_neighbors
