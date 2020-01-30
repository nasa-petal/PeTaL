# Editor: Lucas Saldyt

from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer

from string import punctuation
from contractions import fix_word as expand_contractions

class Cleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer    = PorterStemmer()

    def clean(self, doc):
        for word in word_tokenize(doc):
            words = expand_contractions(word)
            for word in words:
                if word not in self.stop_words and word not in punctuation:
                    yield self.stemmer.stem(word)
