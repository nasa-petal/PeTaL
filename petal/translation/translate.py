from wiki_topic_tree import WikipediaTreeScraper, WikipediaTopicTree, WikipediaTopicNode
from pprint import pprint

import pickle, os

def save(data, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile)

def load(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

class TopicTranslator():
    def __init__(self, *topics):
        self.topic_trees = dict()
        self.scraper = WikipediaTreeScraper()
        for topic in topics:
            topic_name = topic.split('.')[0]
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
        target_results = target_tree.query(term)
        # pprint(source_results)
        # pprint(target_results)

        joint_terms = set.union(*(s for s in source_results.values()), *(s for s in target_results.values()))
        pprint(joint_terms)

        # combined_eng = set.union(*(s for s in results[0].values()))
        # results = bio_tree.query('separate')
        # combined_bio = set.union(*(s for s in results[0].values()))
        # pprint(set.intersection(combined_bio, combined_eng))
        

if __name__ == '__main__':
    translator = TopicTranslator('biology.n.01', 'engineering.n.02')
    translator.translate('branch', 'biology', 'engineering')

