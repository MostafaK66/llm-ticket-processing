
from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor
from knowledge_graph_rag import settings
from knowledge_graph_rag.ticket_graph import TicketsGraph
from knowledge_graph_rag.plotting import Plotter


def main():
    text_preprocessor = TextPreprocessor()
    plotter = Plotter()

    preprocessed_tickets = text_preprocessor.remove_stop_words_from_and_lemmatise_tickets(tickets=settings.tickets)
    ticket_graph = TicketsGraph(tickets=preprocessed_tickets)
    plotter.plot_ticket_graph(graph=ticket_graph.G, output_path=settings.OUTPUT_PASS_PLOTTING)
    input_sentence = "Ticket ID: 105, Issue: Printer not responding, Solution: Check the printer connections and restart the print spooler service."
    connected_docs = ticket_graph.find_connected_documents(input_sentence, N=3)
    print("Connected Documents:", connected_docs)

    for doc in preprocessed_tickets:
        print(doc)








if __name__ == "__main__":
    main()
