import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class TextPreprocessor:
    def __init__(self):
        custom_stop_words = set(stopwords.words("english"))
        important_words = {
            "not",
            "down",
            "up",
            "in",
            "out",
            "off",
            "on",
            "over",
            "under",
        }
        self.stop_words = custom_stop_words - important_words
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, sentence):
        ticket_id_match = re.search(r"Ticket ID: (\d+)", sentence, re.IGNORECASE)
        issue_match = re.search(
            r"Issue: (.*?)(?=, Solution:|$)", sentence, re.IGNORECASE
        )
        solution_match = re.search(r"Solution: (.*)", sentence, re.IGNORECASE)

        ticket_id = ticket_id_match[0] if ticket_id_match else ""
        issue_text = issue_match[1] if issue_match else ""
        solution_text = solution_match[1] if solution_match else ""

        filtered_issue_words = self.filter_and_lemmatize(issue_text)
        filtered_solution_words = self.filter_and_lemmatize(solution_text)

        return self.construct_preprocessed_sentence(
            ticket_id, filtered_issue_words, filtered_solution_words
        )

    def filter_and_lemmatize(self, text):
        words = word_tokenize(text)
        return [
            self.lemmatizer.lemmatize(word.lower())
            for word in words
            if self.is_valid_word(word)
        ]

    def is_valid_word(self, word):
        return (
            any(char.isalpha() for char in word) and word.lower() not in self.stop_words
        )

    def construct_preprocessed_sentence(self, ticket_id, issue_words, solution_words):
        issue_part = "issue " + " ".join(issue_words)
        solution_part = "solution " + " ".join(solution_words)
        ticket_id_part = "Ticket ID" if ticket_id else ""
        if ticket_id_part:
            return f"{ticket_id_part}, {issue_part} {solution_part}"
        return f"{issue_part} {solution_part}"

    def remove_stop_words_from_and_lemmatise_tickets(self, tickets):
        return [self.preprocess_text(doc) for doc in tickets]
