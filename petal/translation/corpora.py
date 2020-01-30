from nltk.corpus import wordnet as wn
from pprint      import pprint

import wikipedia
import pickle, os

from cleaner import Cleaner

class WikipediaTopicNode():
    def __init__(self, synnode, cleaner, parent=None):
        print('Creating node for ', synnode, flush=True)
        self.name = (synnode.name().split('.')[0])
        self.children = []
        self.parent = parent
        try:
            self.page = wikipedia.page(wikipedia.search(self.name)[0], auto_suggest=False)
            self.summary_text = list(cleaner.clean(self.page.summary))
        except wikipedia.exceptions.DisambiguationError:
            self.summary_text = ''

class WikipediaTreeScraper():
    def __init__(self):
        self.cleaner = Cleaner()
        self.hypo = lambda x : x.hyponyms()

    def corpus_tree(self, topic):
        root         = wn.synset(topic)
        name         = (root.name().split('.')[0])
        hyponym_tree = root.tree(self.hypo)
        pprint(hyponym_tree)
        return self.create_tree(hyponym_tree)

    def create_tree(self, hyponym_tree, parent=None):
        if len(hyponym_tree) == 1:
            root = hyponym_tree[0]
            return WikipediaTopicNode(root, self.cleaner, parent=parent)
        root, *children = hyponym_tree
        root = WikipediaTopicNode(root, self.cleaner)
        root.children = [self.create_tree(subtopic, parent=root) for subtopic in children]
        return root

def save(data, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)

def load(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

def cache_corpus_tree(name, filename):
    if not os.path.isfile(filename):
        tree = scraper.corpus_tree(name)
        save(tree, filename)
    else:
        tree = load(filename)
    return tree

if __name__ == '__main__':
    bio = 'biology.n.01'
    eng = 'engineering.n.02'

    scraper  = WikipediaTreeScraper()
    bio_tree = cache_corpus_tree(bio, 'biotree.pkl')
    eng_tree = cache_corpus_tree(eng, 'engtree.pkl')
    bio_set  = set((bio_tree.summary_text))
    eng_set  = set((eng_tree.summary_text))
    print(set.intersection(bio_set, eng_set))

