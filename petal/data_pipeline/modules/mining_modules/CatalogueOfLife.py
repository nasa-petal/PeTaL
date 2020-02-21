from pprint import pprint
from subprocess import call
from time import time

import requests, zipfile, os
from uuid import uuid4

from ..utils.module import Module

'''
This is the backbone mining module for population neo4j with the initial species list
'''


def create_dir():
    # TODO: setup auto downloads from here by scraping most recent date?
    col_date = '2019-05-01' # Make sure this is a valid COL release
    if not os.path.isfile('data/.col_data/taxa.txt'):
        try:
            data = requests.get('http://www.catalogueoflife.org/DCA_Export/zip-fixed/{}-archive-complete.zip'.format(col_date))
            with open('col.zip', 'wb') as outfile:
                outfile.write(data.content)
            with zipfile.ZipFile('col.zip', 'r') as zip_handle:
                zip_handle.extractall('data/.col_data')
        except:
            if os.path.isfile('col.zip'):
                os.remove('col.zip')
            shutil.rmtree('data/.col_data')

class CatalogueOfLife(Module):
    '''
    This module populates neo4j with Species nodes, allowing WikipediaModule and others to process them.
    Notice how BackboneModule's in_label is None, which specifies that it is independent of other neo4j nodes
    '''
    def __init__(self, in_label=None, out_label='Species:Taxon', connect_label=None, name='COL', count=2700000):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self):
        '''
        All that this function does is yield Transaction() objects which create Species() nodes in the neo4j database.
        This particular process() function is simply downloading a tab-separated file and parsing it.
        '''
        create_dir() # Call the code above to download COL data if it isn't already present
        start = time()
        i = 0
        with open('data/.col_data/taxa.txt', 'r', encoding='utf-8') as infile:
            headers = None
            json    = dict()
            # Parse lines of the downloaded file, and add it as a default_transaction() (see yield statement)
            for line in infile:
                if i == 0:
                    headers = line.split('\t')
                    headers = ('id',) + tuple(headers[1:])
                else:
                    for k, v in zip(headers, line.split('\t')):
                        json[k] = v
                    try:
                        json.pop('isExtinct\n')
                    except KeyError:
                        pass
                    if json['taxonRank'] == 'species':
                        json['name'] = json['scientificName'].replace(json['scientificNameAuthorship'], '').strip()
                        json['uuid'] = str(uuid4())
                        yield self.default_transaction(json) # HERE is where the transaction is created!!
                        last_label = self.out_label
                        uuid       = json['uuid']
                        for taxon in ['subgenus', 'genus', 'family', 'superfamily', 'order', 'class', 'phylum', 'kingdom']:
                            label_name = taxon[0].upper() + taxon[1:] + ':Taxon'
                            name = json[taxon]
                            if name == '':
                                continue
                            data = dict(name=name, uuid=name)
                            yield self.custom_transaction(data=data, in_label=last_label, out_label=label_name, connect_labels=('taxon_rank', 'taxon_rank'), uuid=uuid)
                            last_label = label_name
                            uuid = name
                    json = dict()
                i += 1
