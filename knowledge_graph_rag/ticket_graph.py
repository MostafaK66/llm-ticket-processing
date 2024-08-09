import os
import shutil

import certifi
import chromadb
import networkx as nx
import numpy as np
import openai
import requests
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class ResponseGenerator:
    def __init__(self, transformer_model):
        self.transformer_model = transformer_model
        self.client = OpenAI(api_key=api_key)

    def get_embedding_batch(self, input_array):
        session = requests.Session()
        session.verify = False
        openai.api_key = api_key
        openai.requestssession = session
        response = self.client.embeddings.create(
            input=input_array, model=self.transformer_model
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

    def get_embeddings(self, tickets):
        return self.get_embedding_batch(tickets)

    def generate_vectors_collection(self, tickets):
        embeddings = self.get_embedding_batch(tickets)
        vectors_collection = [
            {ticket: embedding} for ticket, embedding in zip(tickets, embeddings)
        ]
        return embeddings, vectors_collection

    def clean_directory_except_sqlite(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename != "chroma.sqlite3":
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    def store_vectors_in_db(self, embeddings, tickets, vectordb_name):
        vectordb_path = os.path.join(vectordb_name)

        self.clean_directory_except_sqlite(vectordb_path)

        client = chromadb.PersistentClient(path=vectordb_name)

        try:
            client.delete_collection(vectordb_name)
        except Exception as e:
            print(f"Failed to delete collection {vectordb_name}. Reason: {e}")

        collection = client.create_collection(vectordb_name)

        collection.add(
            embeddings=embeddings,
            documents=tickets,
            metadatas=[{"source": ""} for _ in range(len(tickets))],
            ids=list(map(str, range(len(tickets)))),
        )

    def query_embedding(self, input_sentence, vectordb_name):
        query_embedding = self.get_embedding_batch([input_sentence])[0]

        client = chromadb.PersistentClient(path=vectordb_name)
        collection = client.get_collection(vectordb_name)

        query_results = collection.query(
            query_embeddings=[query_embedding], n_results=3
        )
        documents = query_results["documents"][0]
        scores = query_results["distances"][0]

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
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]
