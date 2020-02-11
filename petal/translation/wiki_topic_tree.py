from nltk.corpus import wordnet as wn
from collections import defaultdict
from pprint import pprint

import wikipedia
from cleaner import Cleaner

def build_corpus_synset(text, cleaner):
    syn_map = defaultdict(set)
    for word in cleaner.tokenize(text):
        synsets = wn.synsets(word)
        syn_stems = {cleaner.stem(lemma) for synset in synsets for lemma in synset.lemma_names()}
        syn_map[word].update(syn_stems)
    return syn_map

class WikipediaTopicTree():
    # Allow insertion and deletion etc?
    def __init__(self, root, cleaner):
        self.root = root
        self.name_mapping = dict()
        self.cleaner = cleaner

    def __contains__(self, key):
        return key in self.root

    def build_name_mapping(self, node):
        self.name_mapping[node.name] = node
        for child in node.children:
            self.build_name_mapping(child)

    def get(self, name):
        return self.name_mapping[name]

    def query(self, term, which='both'):
        return self.root.query(term, which=which)

class WikipediaTopicNode():
    def __init__(self, synnode, cleaner, parent=None):
        print('Creating node for ', synnode, flush=True)
        self.name = (synnode.name().split('.')[0])
        self.children = []
        self.parent = parent
        try:
            self.page = wikipedia.page(wikipedia.search(self.name)[0], auto_suggest=False)
            self.summary_map = build_corpus_synset(self.page.summary, cleaner)
            self.content_map = build_corpus_synset(self.page.content, cleaner)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                self.page = wikipedia.page(e.options[0], auto_suggest=False)
            except Exception as e:
                print(e)
            self.summary_map = dict()
            self.content_map = dict()

    def __contains__(self, key):
        if key in self.summary_map or key in self.content_map:
            return True
        else:
            for c in self.children:
                if key in c:
                    return True
        return False


    def query(self, term, which='both'):
        pprint(self.summary_map)
        pprint(self.content_map)
        if which == 'summary':
            syn_map = self.summary_map
        elif which == 'content':
            syn_map = self.content_map
        elif which == 'both':
            syn_map = dict()
            syn_map.update(self.summary_map)
            syn_map.update(self.content_map)
        else:
            raise ValueError('Incorrect "which" argument to WikiTree.query(). Must be "content", "summary", or "both"')
        immediate_matches = {k for k, v in syn_map.items() if term in v}
        if len(immediate_matches) > 0:
            matches = {self.name : immediate_matches}
        else:
            matches = dict()
        for c in self.children:
            matches.update(c.query(term))
        return matches

class WikipediaTreeScraper():
    def __init__(self):
        self.cleaner = Cleaner()
        self.hypo = lambda x : x.hyponyms()

    def corpus_tree(self, topic):
        root         = wn.synset(topic)
        name         = (root.name().split('.')[0])
        hyponym_tree = root.tree(self.hypo)
        return WikipediaTopicTree(self.create_tree(hyponym_tree), self.cleaner)

    def create_tree(self, hyponym_tree, parent=None):
        if len(hyponym_tree) == 1:
            root = hyponym_tree[0]
            return WikipediaTopicNode(root, self.cleaner, parent=parent)
        root, *children = hyponym_tree
        root = WikipediaTopicNode(root, self.cleaner)
        root.children = [self.create_tree(subtopic, parent=root) for subtopic in children]
        return root

