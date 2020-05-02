from bitflow.utils.module import Module
from ..libraries.natural_language.hitlist import HitList

from pprint import pprint
from collections import defaultdict
from bisect import bisect_left

import pickle, os

LEXICON_PATH = 'data/lexicon'
INDEX_PATH   = 'data/index'

def insort(items, item, key=lambda x : x):
    '''
    Inplace sorting for items.
    Technically O(n), since it needs to compute keys.
    Could be O(lg(n)) if this didn't happen.
    '''
    keys  = [key(x) for x in items]
    k     = key(item)
    index = bisect_left(keys, k)
    items.insert(index, item)

class InvertedIndexCreator(Module):
    '''
    Create an inverted index from hitlists..

    Used for text search.
    '''
    def __init__(self, in_label='HitList', out_label=None, connect_labels=None, name='InvertedIndexCreator'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=True)

        self.index   = defaultdict(list)
        self.lexicon = set()

    def save(self):
        with open(INDEX_PATH, 'wb') as outfile:
            pickle.dump(self.index, outfile)
        with open(LEXICON_PATH, 'wb') as outfile:
            pickle.dump(self.lexicon, outfile)

    def load(self):
        if os.path.isfile(INDEX_PATH):
            with open(INDEX_PATH, 'rb') as infile:
                self.index = pickle.load(infile)
        if os.path.isfile(LEXICON_PATH):
            with open(LEXICON_PATH, 'rb') as infile:
                self.lexicon = pickle.load(infile)

    def rank(self, entry):
        '''
        This is the custom ranking function, replaceable by any ranking function..
        '''
        *hits, uuid = entry
        print(hits, uuid, sum(hits), flush=True)
        return sum(hits) # TODO replace with custom ranking function with weights

    def process(self, previous):
        '''
        Process a single hitlist.
        Simply sort the inverted index by keyword matches, using self.rank()

        :param previous: neo4j transaction representing a hitlist of an article
        '''
        data = previous.data
        hitlist = HitList(data['source_uuid'])
        try:
            hitlist.load()
            for word in hitlist.words:
                self.lexicon.add(word)

                sections, counts = hitlist.word_hitlist(word)
                insort(self.index[word], (counts + (hitlist.uuid,)), key=lambda x : -self.rank(x))
        except OSError as e:
            print(e)

    def process_batch(self, batch):
        '''
        Process a batch of HitLists

        :param batch: The batch of hitlists..
        '''
        print('BATCH: ', batch.uuid, flush=True)
        self.load()
        for item in batch.items:
            self.process(item)
            print('    UUID: ', item.uuid, flush=True)
        self.save()
        print('')
