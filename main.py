
from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor
from knowledge_graph_rag import settings
from knowledge_graph_rag.ticket_graph import TicketsGraph
from knowledge_graph_rag.plotting import Plotter


def main():
    text_preprocessor = TextPreprocessor()
    plotter = Plotter()

    preprocessed_tickets = text_preprocessor.remove_stop_words_from_and_lemmatise_tickets(tickets=settings.TICKETS)
    ticket_graph = TicketsGraph(tickets=settings.TICKETS)
    plotter.plot_ticket_graph(graph=ticket_graph.G, output_path=settings.OUTPUT_PASS_PLOTTING)
    input_sentence = "Ticket ID: 116, Issue: Printer not responding, Solution: Check the printer connections and restart the print spooler service."
    connected_docs = ticket_graph.find_connected_documents(input_sentence, n = settings.NUMBER_OF_SIMILAR_TICKETS)
    out_put = ticket_graph.find_n_similar_tickets(input_sentence, n = settings.NUMBER_OF_SIMILAR_TICKETS)
    print("yes")










if __name__ == "__main__":
    main()
