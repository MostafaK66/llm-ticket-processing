import os
import openai
import certifi

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
        return vectors_collection
