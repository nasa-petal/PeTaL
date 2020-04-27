from scholarly import search_pubs_query as google_scholar_search

from petal.pipeline.module_utils.module import Module

from random import random, randint
from pprint import pprint

class GoogleScholarModule(Module):
    def __init__(self, in_label='Taxon', out_label='GoogleArticle:Article', connect_labels=('MENTIONED_IN_ARTICLE', 'MENTIONS_SPECIES'), name='GoogleScholar'):
        Module.__init__(self, in_label, out_label, connect_labels, name)


    def process(self, previous):
        name = previous.data['name']
        scholar_result_gen = google_scholar_search(name)
        limit = randint(5, 20)
        results = []
        for i in range(limit):
            data = next(scholar_result_gen).bib
            data['content'] = ''
            results.append(self.default_transaction(data, uuid=data['title'] + '_GoogleArticle', from_uuid=previous.data['uuid']))
        return results
