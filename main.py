from knowledge_graph_rag import settings
from knowledge_graph_rag.ticket_graph import TicketsGraph
from knowledge_graph_rag.plotting import Plotter


def main():
    plotter = Plotter()
    ticket_graph = TicketsGraph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=ticket_graph.G, output_path=settings.OUTPUT_PASS_PLOTTING)
    input_sentence = "Ticket ID: 116, Issue: Printer not responding"
    out_put = ticket_graph.find_n_similar_tickets(input_sentence, n=settings.NUMBER_OF_SIMILAR_TICKETS)
    print(out_put)


if __name__ == "__main__":
    main()
