from wiki_topic_tree import WikipediaTreeScraper, WikipediaTopicTree, WikipediaTopicNode
from nltk.corpus import wordnet as wn
from pprint import pprint

import pickle, os

def save(data, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)

def load(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

def flatten_tree(tree, leaf_set):
    leaf_set.add(tree[0].name().split('.')[0])
    if len(tree) == 1:
        return
    else:
        root, *children = tree
        for child in children:
            flatten_tree(child, leaf_set)

hyp = lambda s : s.hyponyms()

class TopicTranslator():
    def __init__(self, *topics):
        self.topic_trees = dict()
        self.synsets = dict()
        self.scraper = WikipediaTreeScraper()
        for topic in topics:
            topic_name = topic.split('.')[0]
            self.synsets[topic_name] = wn.synset(topic)
            self.topic_trees[topic_name] = self.cache_corpus_tree(topic, 'topic_tree_cache/{}.pkl'.format(topic))

    def cache_corpus_tree(self, name, filename):
        if not os.path.isfile(filename):
            tree = self.scraper.corpus_tree(name)
            save(tree, filename)
        else:
            tree = load(filename)
        return tree

    def translate(self, term, source, target):
        source_tree = self.topic_trees[source]
        target_tree = self.topic_trees[target]

        source_results = source_tree.query(term)

        results = {term}
        for source_syn in source_results.values():
            target_results = target_tree.query(term)
            results.update(value for s in target_results.values() for value in s)
        return results

if __name__ == '__main__':
    translator = TopicTranslator('biology.n.01', 'engineering.n.02')
    terms = {'branch', 'channel', 'connect', 'couple', 'control', 'convert', 'provision', 'signal', 'support', 'give', 'move'}
    # terms = {'signal', 'sense', 'detect'}
    for term in terms:
        print(term)
        results = translator.translate(term, 'biology', 'engineering')
        pprint(results)

