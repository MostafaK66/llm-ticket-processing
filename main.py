from knowledge_graph_rag import settings
from knowledge_graph_rag.ticket_graph import TicketsGraph
from knowledge_graph_rag.plotting import Plotter
from knowledge_graph_rag.generate_response import ResponseGenerator


def main():
    plotter = Plotter()
    generator = ResponseGenerator(api_key=settings.OPENAI_API_KEY, transformer_model=settings.TRANSFORMER_MODEL)
    ticket_graph = TicketsGraph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=ticket_graph.G, output_path=settings.OUTPUT_PASS_PLOTTING)
    input_sentence = "Ticket ID: 116, Issue: Printer not responding"
    out_put = ticket_graph.find_n_similar_tickets(input_sentence, n=settings.NUMBER_OF_SIMILAR_TICKETS)
    vectors_collection = generator.generate_vectors_collection(settings.TICKETS)
    for vector in vectors_collection:
        print(vector)


if __name__ == "__main__":
    main()
