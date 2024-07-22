import os
import openai
import certifi
import chromadb
import shutil
import networkx as nx
import numpy as np
os.environ['SSL_CERT_FILE'] = certifi.where()


class ResponseGenerator:
    def __init__(self, api_key, transformer_model):
        openai.api_key = api_key
        self.transformer_model = transformer_model

    def get_embedding_batch(self, input_array):
        response = openai.Embedding.create(
            input=input_array,
            model=self.transformer_model
        )
        return [data['embedding'] for data in response['data']]

    def generate_vectors_collection(self, tickets):
        embeddings = self.get_embedding_batch(tickets)
        vectors_collection = [{ticket: embedding} for ticket, embedding in zip(tickets, embeddings)]
        return embeddings, vectors_collection

    def clean_directory_except_sqlite(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename != 'chroma.sqlite3':
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def store_vectors_in_db(self, embeddings, tickets, vectordb_name):
        vectordb_path = os.path.join(vectordb_name)

        self.clean_directory_except_sqlite(vectordb_path)

        client = chromadb.PersistentClient(path=vectordb_name)

        try:
            client.delete_collection(vectordb_name)
        except Exception as e:
            print(f'Failed to delete collection {vectordb_name}. Reason: {e}')

        collection = client.create_collection(vectordb_name)

        collection.add(
            embeddings=embeddings,
            documents=tickets,
            metadatas=[{"source": ""} for _ in range(len(tickets))],
            ids=list(map(str, range(len(tickets))))
        )

    def query_embedding(self, input_sentence, vectordb_name):
        query_embedding = self.get_embedding_batch([input_sentence])[0]

        client = chromadb.PersistentClient(path=vectordb_name)
        collection = client.get_collection(vectordb_name)

        query_results = collection.query(query_embeddings=[query_embedding], n_results=3)
        documents = query_results['documents'][0]
        scores = query_results['distances'][0]

        return list(zip(documents, scores))

    def create_graph(self, tickets):
        embeddings = self.get_embedding_batch(tickets)
        num_tickets = len(tickets)

        G = nx.Graph()

        for i, ticket in enumerate(tickets):
            G.add_node(i, label=ticket)

        for i in range(num_tickets):
            for j in range(i + 1, num_tickets):
                score = self.calculate_similarity(embeddings[i], embeddings[j])
                if score > 0:
                    G.add_edge(i, j, weight=score)

        return G

    def calculate_similarity(self, embedding1, embedding2):
        return np.linalg.norm(np.array(embedding1) - np.array(embedding2))



