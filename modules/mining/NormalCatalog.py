from pprint import pprint
from subprocess import call
from time import time

import requests, zipfile, os

from bitflow.utils.module import Module

'''
This is the backbone mining module for population neo4j with the initial species list

Inefficient compared to bulk LOAD CSV imports.
'''

def create_dir():
    '''
    Download the most recent taxon catalog
    '''
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
            if rank == 'species':
                json['name'] = json['scientificName'].replace(json['scientificNameAuthorship'], '').strip()
            elif rank == 'infraspecies':
                continue
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
    def __init__(self, in_label=None, out_label='Species:Taxon', connect_label=None, name='NormalCatalog'):
        Module.__init__(self, in_label, out_label, connect_label, name)

    def process(self):
        '''
        All that this function does is yield Transaction() objects which create Species() nodes in the neo4j database.
        This particular process() function is simply downloading a tab-separated file and parsing it.
        '''
        print('Running catalog', flush=True)
        created_edges = set()
        created_nodes = []

        for json in to_json():

            ### Adds species nodes and creates transaction that will be accessed by any dependent modules
            rank = json['taxonRank']
            if rank == 'species' and json['name'] not in created_nodes:       # Check to make sure rank is species, not a higher taxonomy category.
                yield self.default_transaction(json, uuid=json['name'])       # HERE is where the transaction is created!! Since this is a default transaction, the label is what is defined in the __init__ function.

            ### Adds non-species taxonomic categories as nodes. Creates edges between all connected taxonomy categories to create a tree of life.
            last_uuid = json['name']
            last_label = rank[0].upper() + rank[1:] + ':Taxon'
            taxon_dict = {'species': ['genus', 'family'], 'genus': ['family', 'superfamily'], 'family': ['superfamily', 'order'], 'superfamily' : ['order', 'class'], 'order': ['class', 'phylum'], 'class': ['phylum', 'kingdom'], 'phylum': ['kingdom'], 'kingdom':[]}
            taxons = taxon_dict[rank]
            for taxon in taxons:
                uuid = json[taxon]  # Creates an unique id for each node (name of the taxon)
                if uuid.strip() == '':
                    continue
                if (last_uuid, uuid) not in created_edges:
                    node = dict(name=uuid, uuid=uuid)  # data for a new node
                    label = taxon[0].upper() + taxon[1:] + ':Taxon'
                    created_edges.add((last_uuid, uuid))

                    if uuid not in created_nodes:  # If the node does not exist yet, create the node and the edge betweeen the related node.
                        yield self.custom_transaction(data=node, in_label=last_label, out_label=label, connect_labels=('supertaxon', 'subtaxon'), uuid=uuid, from_uuid=last_uuid)
                    else: # If the node exists already, just create the edge between the related nodes.
                        yield self.custom_transaction(in_label=last_label, out_label=label, connect_labels=('supertaxon', 'subtaxon'), uuid=uuid, from_uuid=last_uuid)
                    created_nodes.append(uuid)

                last_label = label
                last_uuid = uuid
