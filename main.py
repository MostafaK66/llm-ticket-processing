

from knowledge_graph_rag.ticket_preprocessing import TextPreprocessor
from knowledge_graph_rag import settings
from knowledge_graph_rag.ticket_graph import TicketsGraph


def main():
    text_preprocessor = TextPreprocessor()

    preprocessed_tickets = text_preprocessor.remove_stop_words_from_and_lemmatise_tickets(settings.tickets)
    ticket_graph = TicketsGraph(settings.tickets)
    ticket_graph.plot_ticket_graph()

    for doc in preprocessed_tickets:
        print(doc)


if __name__ == "__main__":
    main()
