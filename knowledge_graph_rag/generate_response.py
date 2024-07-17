import os
import openai
import certifi
import chromadb
import shutil

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

    def store_vectors_in_db(self, embeddings, tickets, vectordb_name="cvd_vectors"):
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

