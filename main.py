

from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor
from knowledge_graph_rag import settings
from knowledge_graph_rag.ticket_graph import TicketsGraph
from knowledge_graph_rag.plotting import Plotter


def main():
    text_preprocessor = TextPreprocessor()
    plotter = Plotter()

    preprocessed_tickets = text_preprocessor.remove_stop_words_from_and_lemmatise_tickets(tickets=settings.tickets)
    ticket_graph = TicketsGraph(tickets=settings.tickets)
    plotter.plot_ticket_graph(graph=ticket_graph.G, output_path=settings.OUTPUT_PASS_PLOTTING)





    for doc in preprocessed_tickets:
        print(doc)


if __name__ == "__main__":
    main()
