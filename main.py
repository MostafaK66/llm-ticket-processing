

from knowledge_graph_rag.text_preprocessing import TextPreprocessor
from knowledge_graph_rag import settings


def main():
    text_preprocessor = TextPreprocessor()

    preprocessed_documents = text_preprocessor.remove_stop_words_from_and_lemmatise_tickets(settings.tickets)

    for doc in preprocessed_documents:
        print(doc)


if __name__ == "__main__":
    main()
