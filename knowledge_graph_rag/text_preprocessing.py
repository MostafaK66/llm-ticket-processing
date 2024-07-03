

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def remove_stop_words_from_and_lemmatise_documents(self, documents):
        def preprocess_text(sentences):
            preprocessed_sentences = []

            for sentence in sentences:

                parts = sentence.split(", ")
                ticket_id = parts[0] if parts[0].lower().startswith("ticket id") else ""
                remaining_text = ", ".join(parts[1:]) if ticket_id else sentence

                words = word_tokenize(remaining_text)

                filtered_words = [
                    self.lemmatizer.lemmatize(word.lower())
                    for word in words
                    if word.lower() not in self.stop_words and word.isalpha()
                ]


                preprocessed_sentence = f"{ticket_id}, " + " ".join(filtered_words) if ticket_id else " ".join(filtered_words)
                preprocessed_sentences.append(preprocessed_sentence)

            return preprocessed_sentences

        preprocessed_documents = preprocess_text(documents)
        return preprocessed_documents
