from scholarly import search_pubs_query as google_scholar_search

from bitflow.utils.module import Module

from random import random, randint
from pprint import pprint

'''
WARNING: Some of the following code disobeys robots.txt of particular websites.
  Thus, it is disabled by default and only left as reference.
  Use at your own risk.

       o     O
     __|_____|___
    |    --      |
    |  ( o )   ( o )
  { |        /   |
    |     [wwww]  < *Exterminate all humans.txt* )
    [____________|
       |   |              /Vvvv/
  _____|___|____          |___/
 /______________\_________/   |
 |              |             /
 | ( / )  ( + ) |__|__|__|_|_/
 |              |
 | [ -vV--vV-]  |
 |              |
 |______________/
'''

class GoogleScholarModule(Module):
    def __init__(self, in_label='Taxon', out_label='GoogleArticle:Article', connect_labels=('MENTIONED_IN_ARTICLE', 'MENTIONS_SPECIES'), name='GoogleScholar'):
        Module.__init__(self, in_label, out_label, connect_labels, name)


    def process(self, previous):
        '''
        Use google scholar search to get articles.
        This data is very valuable, but is technically not allowed to be scraped.
        Thus, this module is disabled by default.

        :param previous: Query to search for
        '''
        name = previous.data['name']
        scholar_result_gen = google_scholar_search(name)
        limit = randint(5, 20)
        results = []
        for i in range(limit):
            data = next(scholar_result_gen).bib
            data['content'] = ''
            results.append(self.default_transaction(data, uuid=data['title'] + '_GoogleArticle', from_uuid=previous.data['uuid']))
        return results
