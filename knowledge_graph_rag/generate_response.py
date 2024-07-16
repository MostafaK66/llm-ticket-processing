import os
import openai
import certifi
import chromadb

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

    def store_vectors_in_db(self, embeddings, tickets, vectordb_name="cvd_vectors"):
        client = chromadb.PersistentClient(path=vectordb_name)
        client.delete_collection("cvd_vectors")
        collection = client.create_collection(vectordb_name)

        collection.add(
            embeddings=embeddings,
            documents=tickets,
            metadatas=[{"source": ""} for _ in range(len(tickets))],
            ids=list(map(str, range(len(tickets))))
        )

