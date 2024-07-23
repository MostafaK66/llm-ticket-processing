from knowledge_graph_rag import settings
from knowledge_graph_rag.plotting import Plotter
from knowledge_graph_rag.ticket_graph import ResponseGenerator
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_KEY')
if not api_key:
    raise ValueError("API key is not set. Please ensure the .env file contains the 'OPENAI_KEY' variable.")


def main():
    plotter = Plotter()
    generator = ResponseGenerator(api_key=api_key, transformer_model=settings.TRANSFORMER_MODEL)
    graph = generator.create_graph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=graph, output_path=settings.OUTPUT_PASS_PLOTTING)
    embeddings, vectors_collection = generator.generate_vectors_collection(settings.TICKETS)
    generator.store_vectors_in_db(embeddings=embeddings, tickets=settings.TICKETS, vectordb_name=settings.VECTORDB_NAME)
    search_results = generator.query_embedding(input_sentence=settings.INPUT_SENTENCE,
                                               vectordb_name=settings.VECTORDB_NAME)
    for idx, result in enumerate(search_results):
        print(f"Search Result {idx + 1}: {result}")


if __name__ == "__main__":
    main()
