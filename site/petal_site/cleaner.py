# Editor: Lucas Saldyt

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.corpus   import wordnet as wn
from nltk.stem     import PorterStemmer

from string import punctuation
from .contractions import fix_word as expand_contractions

class Cleaner:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            import nltk
            nltk.download('stopwords')
        self.stemmer    = PorterStemmer()

    def tokenize(self, text):
        return word_tokenize(text)

    def clean(self, doc, doc_as_words=False):
        if doc_as_words:
            words = doc
        else:
            words = word_tokenize(doc)

        for word in words:
            for word in self.clean_word(word):
                yield word
                # yield self.stem(word)

    def clean_word(self, word):
        words = expand_contractions(word)
        for word in words:
            if word not in self.stop_words and word not in punctuation:
                yield word

    def stem(self, word):
        return self.stemmer.stem(word)

