# from knowledge_graph_rag import settings
# from knowledge_graph_rag.plotting import Plotter
# from knowledge_graph_rag.ticket_graph import ResponseGenerator
# from knowledge_graph_rag.knowledge_graph import KnowledgeGraphGenerator
# import os
# from dotenv import load_dotenv
# import certifi
#
# load_dotenv()
#
# os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
#
# api_key = os.getenv('OPENAI_API_KEY')
# if not api_key:
#     raise ValueError("API key is not set. Please ensure the .env file contains the 'OPENAI_API_KEY' variable.")
#
#
# def main():
#     plotter = Plotter()
#     generator = ResponseGenerator(transformer_model=settings.TRANSFORMER_MODEL)
#     graph = generator.create_graph(tickets=settings.TICKETS)
#     plotter.plot_ticket_graph(graph=graph, output_path=settings.OUTPUT_PASS_PLOTTING)
#     embeddings, vectors_collection = generator.generate_vectors_collection(settings.TICKETS)
#     generator.store_vectors_in_db(embeddings=embeddings, tickets=settings.TICKETS, vectordb_name=settings.VECTORDB_NAME)
#     search_results = generator.query_embedding(input_sentence=settings.INPUT_SENTENCE,
#                                                vectordb_name=settings.VECTORDB_NAME)
#     # for idx, result in enumerate(search_results):
#     #     print(f"Search Result {idx + 1}: {result}")
#
#     # Generate knowledge representations
#     kg_generator = KnowledgeGraphGenerator()
#     knowledge_representations = kg_generator.create_knowledge_representations(tickets=settings.TICKETS)
#     # print("Knowledge Graph Representations:")
#     print(knowledge_representations)
#     kn_graph = kg_generator.create_knowledge_graph_from_representations(representations=knowledge_representations)
#     plotter.plot_kn_graph(graph=kn_graph, output_path="outputs/kn_graph.png")
#     search_result = kg_generator.search_ticket(input_ticket=settings.INPUT_SENTENCE, max_depth=settings.MAX_SEARCH_DEPTH)
#     # print("Search Result:")
#     # print(search_result)
#
#
# if __name__ == "__main__":
#     main()
from knowledge_graph_rag import settings
from knowledge_graph_rag.plotting import Plotter
from knowledge_graph_rag.ticket_graph import ResponseGenerator
from knowledge_graph_rag.knowledge_graph import KnowledgeGraphGenerator
import os
from dotenv import load_dotenv
import certifi

load_dotenv()

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key is not set. Please ensure the .env file contains the 'OPENAI_API_KEY' variable.")


def main():
    plotter = Plotter()
    generator = ResponseGenerator(transformer_model=settings.TRANSFORMER_MODEL)
    kg_generator = KnowledgeGraphGenerator()
    graph = generator.create_graph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=graph, output_path=settings.OUTPUT_PASS_PLOTTING)
    embeddings, vectors_collection = generator.generate_vectors_collection(settings.TICKETS)
    generator.store_vectors_in_db(embeddings=embeddings, tickets=settings.TICKETS, vectordb_name=settings.VECTORDB_NAME)

    knowledge_representations = kg_generator.create_knowledge_representations(tickets=settings.TICKETS)

    kn_graph = kg_generator.create_knowledge_graph_from_representations(representations=knowledge_representations)

    # Generate and integrate embeddings
    # embeddings = generator.get_embedding_batch(settings.TICKETS)
    kg_generator.integrate_embeddings(settings.TICKETS, embeddings)

    plotter.plot_kn_graph(graph=kn_graph, output_path="outputs/kn_graph.png")

    # Generate embedding for the input ticket
    input_embedding = generator.get_embedding_batch([settings.INPUT_SENTENCE])[0]
    search_result = kg_generator.search_ticket(input_ticket=settings.INPUT_SENTENCE, input_embedding=input_embedding,
                                               max_depth=3)
    print("Search Result:")
    print(search_result)


if __name__ == "__main__":
    main()

