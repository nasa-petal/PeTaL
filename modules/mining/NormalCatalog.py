from pprint import pprint
from subprocess import call
from time import time

import requests, zipfile, os

from petal.pipeline.module_utils.module import Module

'''
This is the backbone mining module for population neo4j with the initial species list
'''

def create_dir():
    # TODO: setup auto downloads from here by scraping most recent date?
    col_date = '2019-05-01' # Make sure this is a valid COL release
    if not os.path.isfile('data/.col_data/taxa.txt'):
        try:
            print('Downloading catalogue of life data (~250mb)', flush=True)
            data = requests.get('http://www.catalogueoflife.org/DCA_Export/zip-fixed/{}-archive-complete.zip'.format(col_date))
            with open('col.zip', 'wb') as outfile:
                outfile.write(data.content)
            print('Done', flush=True)
            with zipfile.ZipFile('col.zip', 'r') as zip_handle:
                zip_handle.extractall('data/.col_data')
            print('Finished')
        except:
            if os.path.isfile('col.zip'):
                os.remove('col.zip')
            shutil.rmtree('data/.col_data')

def to_json():
    '''
    All that this function does is yield Transaction() objects which create Species() nodes in the neo4j database.
    This particular process() function is simply downloading a tab-separated file and parsing it.
    '''
    create_dir() # Call the code above to download COL data if it isn't already present
    start = time()
    with open('data/.col_data/taxa.txt', 'r', encoding='utf-8') as infile:
        json    = dict()
        # Parse lines of the downloaded file, and add it as a default_transaction() (see yield statement)
        headers = infile.readline().split('\t')
        headers = ('id',) + tuple(headers[1:])
        for line in infile:
            for k, v in zip(headers, line.split('\t')):
                json[k] = v.replace('"', '')
            try:
                json.pop('isExtinct\n')
            except KeyError:
                pass
            rank = json['taxonRank']
            if rank == 'species' or rank == 'infraspecies':
                json['name'] = json['scientificName'].replace(json['source'], '').strip()
            else:
                json['name'] = json[rank]
            if json['name'] == 'Not assigned':
                continue
            yield json
            json = dict()


class NormalCatalog(Module):
    '''
    This module populates neo4j with Species nodes, allowing WikipediaModule and others to process them.
    Notice how BackboneModule's in_label is None, which specifies that it is independent of other neo4j nodes
    '''
    def __init__(self, in_label=None, out_label='Species:Taxon', connect_label=None, name='NormalCatalog', count=2700000):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self):
        '''
        All that this function does is yield Transaction() objects which create Species() nodes in the neo4j database.
        This particular process() function is simply downloading a tab-separated file and parsing it.
        '''
        print('Running catalog', flush=True)
        seen = set()
        for json in to_json():
            yield self.default_transaction(json, uuid=json['name']) # HERE is where the transaction is created!!
            last_uuid = json['name']
            last_label = 'Species:Taxon'
            for taxon in ['subgenus', 'genus', 'family', 'superfamily', 'order', 'class', 'phylum', 'kingdom']:
                name = json[taxon]
                if name.strip() == '':
                    continue
                if (last_uuid, name) not in seen:
                    data = dict(name=name, uuid=name)
                    label_name = taxon[0].upper() + taxon[1:] + ':Taxon'
                    seen.add((last_uuid, name))
                    yield self.custom_transaction(data=data, in_label=last_label, out_label=label_name, connect_labels=('supertaxon', 'subtaxon'), uuid=name, from_uuid=last_uuid)
                last_label = label_name
                last_uuid = name
