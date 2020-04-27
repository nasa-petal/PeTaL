from pprint import pprint
from subprocess import call
from time import time, sleep

import requests, zipfile, os
from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

import neobolt

from petal.pipeline.module_utils.module import Module

from .NormalCatalog import create_dir, to_json

IMPORT = '../../.Neo4jDesktop/neo4jDatabases/database-f009728a-c309-4d9d-937d-cbfd4d57ee42/installation-4.0.2/import/'

'''
This is the backbone mining module for population neo4j with the initial species list
'''

def to_long_json():
    for json in to_json():
        found = False
        relations = []
        for taxon in ['species', 'subgenus', 'genus', 'family', 'superfamily', 'order', 'class', 'phylum', 'kingdom']:
            if found:
                if json[taxon].strip() != '':
                    relations.append((json['name'], json[taxon]))
                    break
            rank = json['taxonRank']
            if rank == 'infraspecies':
                if taxon == 'species': # Necessary
                    found = True
            elif taxon == rank:
                found = True
        yield json, relations

def to_csv():
    i = 0
    first = True
    with open('data/cache/catalog.csv', 'w', encoding='utf-8') as catalog:
        with open('data/cache/species.csv', 'w', encoding='utf-8') as species_csv:
            with open('data/cache/relations.csv', 'w', encoding='utf-8') as relations:
                for entry, rels in to_long_json():
                    if i % 1000 == 0:
                        print(i, flush=True)
                    i += 1
                    if first:
                        catalog.write(','.join(entry.keys()) + '\n')
                        species_csv.write(','.join(entry.keys()) + '\n')
                        relations.write('from,to\n')
                        first = False
                    if entry['taxonRank'] == 'species':
                        species_csv.write(','.join(entry.values()) + '\n')
                    else:
                        catalog.write(','.join(entry.values()) + '\n')
                    if len(rels) > 0:
                        relations.write('\n'.join(','.join(r) for r in rels) + '\n')


class OptimizedCatalog(Module):
    '''
    Populate Taxa into the database in an optimized manor
    '''
    def __init__(self, import_dir=IMPORT, in_label=None, out_label='CatalogFinishedSignal', connect_label=None, name='OptimizedCatalog'):
        Module.__init__(self, in_label, out_label, connect_label, name)
        self.import_dir = import_dir

    def process(self):
        self.driver = self.get_driver(driver=driver)
        if self.driver.get('__optimized_catalog_finished_signal__') is not None:
            return
        if not os.path.isfile('data/cache/catalog.csv') or not os.path.isfile('data/cache/relations.csv') or not os.path.isfile('data/cache/species.csv'):
            to_csv()
        for filename in os.listdir('data/cache/'):
            if filename.endswith('.csv'):
                pathname = 'data/cache/' + filename
                shutil.copy(pathname, self.import_dir + filename)

        with self.driver.neo_client.session() as session:
            with open('data/cache/catalog.csv', 'r') as infile:
                headers = infile.readline().split(',')
            try:
                session.run('CREATE INDEX ON :Taxon(taxonRank)')
            except neobolt.exceptions.ClientError:
                pass
            try:
                session.run('CREATE INDEX ON :Taxon(name)')
            except neobolt.exceptions.ClientError:
                pass
            print('Adding catalog.csv')
            session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///catalog.csv" AS line CREATE (x:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
            print('Adding species.csv')
            session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///species.csv" AS line CREATE (x:Species:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
            print('Adding relations.csv')
            session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///relations.csv" AS line MATCH (x:Taxon {name: line.from}),(y:Taxon {name: line.to}) CREATE (x)-[:supertaxon]->(y)')

        yield self.default_transaction(data=dict(done=True), uuid='__optimized_catalog_finished_signal__')
        print('Optimized Catalog finished')
