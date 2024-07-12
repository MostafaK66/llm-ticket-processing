import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor


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

    def find_n_similar_tickets(self, input_sentence, n):
        input_sentence = self.preprocessor.preprocess_text(input_sentence)

        all_tickets = self.preprocessed_tickets + [input_sentence]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_tickets)
        cosine_sim = cosine_similarity(tfidf_matrix)
        similarity_scores = cosine_sim[-1, :-1]
        closest_indices = np.argsort(similarity_scores)[-n:][::-1]

        return [
            (self.preprocessed_tickets[idx], similarity_scores[idx])
            for idx in closest_indices
            if similarity_scores[idx] > 0
        ]
