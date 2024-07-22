from knowledge_graph_rag import settings
# from knowledge_graph_rag.ticket_graph import TicketsGraph
from knowledge_graph_rag.plotting import Plotter
from knowledge_graph_rag.generate_response import ResponseGenerator


def main():
    plotter = Plotter()
    generator = ResponseGenerator(api_key=settings.OPENAI_API_KEY, transformer_model=settings.TRANSFORMER_MODEL)
    # ticket_graph = TicketsGraph(tickets=settings.TICKETS)
    # plotter.plot_ticket_graph(graph=ticket_graph.G, output_path=settings.OUTPUT_PASS_PLOTTING)
    graph = generator.create_graph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=graph, output_path=settings.OUTPUT_PASS_PLOTTING)
    input_sentence = "Ticket ID: 1188, Issue: something wrong with printer"
    # out_put = ticket_graph.find_n_similar_tickets(input_sentence, n=settings.NUMBER_OF_SIMILAR_TICKETS)
    embeddings, vectors_collection = generator.generate_vectors_collection(settings.TICKETS)

    generator.store_vectors_in_db(embeddings=embeddings, tickets=settings.TICKETS, vectordb_name=settings.VECTORDB_NAME)

    search_results = generator.query_embedding(input_sentence=input_sentence, vectordb_name=settings.VECTORDB_NAME)
    for idx, result in enumerate(search_results):
        print(f"Search Result {idx + 1}: {result}")


if __name__ == "__main__":
    main()
