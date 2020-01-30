from nltk.corpus import wordnet as wn
from pprint      import pprint

import wikipedia

from cleaner import Cleaner

class WikipediaTopicNode():
    def __init__(self, synnode, cleaner):
        print('Creating node for ', synnode, flush=True)
        self.name = (synnode.name().split('.')[0])
        self.children = []
        try:
            self.page = wikipedia.page(wikipedia.search(self.name)[0], auto_suggest=False)
            self.summary_text = cleaner.clean(self.page.summary)
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

    def create_tree(self, hyponym_tree):
        if len(hyponym_tree) == 1:
            root = hyponym_tree[0]
            return WikipediaTopicNode(root, self.cleaner)
        root, *children = hyponym_tree
        root = WikipediaTopicNode(root, self.cleaner)
        root.children = [self.create_tree(subtopic) for subtopic in children]
        return root


if __name__ == '__main__':
    bio = 'biology.n.01'
    eng = 'engineering.n.02'

    scraper = WikipediaTreeScraper()
    print(scraper.corpus_tree(bio))
    print(scraper.corpus_tree(eng))

