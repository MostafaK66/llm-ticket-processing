# text_preprocessing.py

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

                words = word_tokenize(sentence)


                filtered_words = [
                    self.lemmatizer.lemmatize(word.lower())
                    for word in words
                    if word.lower() not in self.stop_words and word.isalpha()
                ]


                preprocessed_sentence = " ".join(filtered_words)
                preprocessed_sentences.append(preprocessed_sentence)

            return preprocessed_sentences


        preprocessed_documents = preprocess_text(documents)
        return preprocessed_documents
