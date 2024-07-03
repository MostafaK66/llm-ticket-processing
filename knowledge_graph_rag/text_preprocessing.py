import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, sentence):
        parts = sentence.split(", ")
        ticket_id = parts[0] if parts[0].lower().startswith("ticket id") else ""

        issue_text, solution_text = self.extract_issue_and_solution(parts)

        filtered_issue_words = self.filter_and_lemmatize(issue_text)
        filtered_solution_words = self.filter_and_lemmatize(solution_text)

        return self.construct_preprocessed_sentence(
            ticket_id, filtered_issue_words, filtered_solution_words
        )

    def extract_issue_and_solution(self, parts):
        issue_text = ""
        solution_text = ""

        for part in parts[1:]:
            if part.lower().startswith("issue:"):
                issue_text = part[7:]
            elif part.lower().startswith("solution:"):
                solution_text = part[10:]

        return issue_text, solution_text

    def filter_and_lemmatize(self, text):
        words = word_tokenize(text)
        return [
            self.lemmatizer.lemmatize(word.lower())
            for word in words
            if self.is_valid_word(word)
        ]

    def is_valid_word(self, word):
        return any(char.isalpha() for char in word) and word.lower() not in self.stop_words

    def construct_preprocessed_sentence(self, ticket_id, issue_words, solution_words):
        issue_part = "issue: " + " ".join(issue_words)
        solution_part = "solution: " + " ".join(solution_words)
        if ticket_id:
            return f"{ticket_id}, {issue_part} {solution_part}"
        return f"{issue_part} {solution_part}"

    def remove_stop_words_from_and_lemmatise_documents(self, documents):
        return [self.preprocess_text(doc) for doc in documents]
