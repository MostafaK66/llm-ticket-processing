from knowledge_graph_rag import settings
from knowledge_graph_rag.plotting import Plotter
from knowledge_graph_rag.ticket_graph import ResponseGenerator

def main():
    plotter = Plotter()
    generator = ResponseGenerator(transformer_model=settings.TRANSFORMER_MODEL)
    graph = generator.create_graph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=graph, output_path=settings.OUTPUT_PASS_PLOTTING)
    embeddings, vectors_collection = generator.generate_vectors_collection(settings.TICKETS)
    generator.store_vectors_in_db(embeddings=embeddings, tickets=settings.TICKETS, vectordb_name=settings.VECTORDB_NAME)
    search_results = generator.query_embedding(input_sentence=settings.INPUT_SENTENCE, vectordb_name=settings.VECTORDB_NAME)
    for idx, result in enumerate(search_results):
        print(f"Search Result {idx + 1}: {result}")

    # Generate knowledge representations
    # kg_generator = KnowledgeGraphGenerator()
    # knowledge_representations = kg_generator.create_knowledge_representations(tickets=settings.TICKETS)
    # print("Knowledge Graph Representations:")
    # print(knowledge_representations)

if __name__ == "__main__":
    main()


