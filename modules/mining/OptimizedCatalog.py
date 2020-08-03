from pprint import pprint
from subprocess import call
from time import time, sleep

import requests, zipfile, os
import csv
from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

import neobolt

from bitflow.utils.module import Module

from .NormalCatalog import create_dir, to_json

IMPORT = '../../.Neo4jDesktop/neo4jDatabases/database-328e805b-80fc-42bc-bf79-44a07e3e1447/installation-4.0.2/import/'

'''
This is the backbone mining module for population neo4j with the initial species list

It is fairly efficient, since it uses native neo4j bulk data importing.

Be sure to update the above IMPORT directory, and even better, automate the discovery of this directory!
This is kept this way to prevent security vulnerabilities when running on a server. (See neo4j LOAD CSV documentation)
'''

def to_long_json():
    '''
    Convert to long form (a list of nodes, and a list of relations between nodes)
    '''
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
    '''
    Transform catalog data to three CSV files that neo4j natively understands.
    species.csv is seperate from catalog.csv to create an individual Species label in the neo4j database
    '''
    i = 0
    first = True
    with open('data/cache/catalog.csv', 'w', encoding='utf-8', newline='') as catalog:
        with open('data/cache/species.csv', 'w', encoding='utf-8', newline='') as species_csv:
            with open('data/cache/relations.csv', 'w', encoding='utf-8', newline='') as relations:
                for entry, rels in to_long_json():
                    if i % 1000 == 0:
                        print(i, flush=True)
                    i += 1
                    
                    output_cat = csv.writer(catalog) #create a csv.write to catalog.csv
                    output_spec = csv.writer(species_csv) #create a csv.write to species.csv
                    output_rels = csv.writer(relations) #create a csv.write to relations.csv
    
                    if first: #Check whether header row has been written. If not, write it.
                        output_cat.writerow(entry.keys())  # header row
                        output_spec.writerow(entry.keys())  # header row
                        output_rels.writerow(['from','to'])  # header row
                        first = False
                        
                    if entry['taxonRank'] == 'species':
                        output_spec.writerow(entry.values())  # enter rows of species
                    else:
                        output_cat.writerow(entry.values())  # enter rows of !species
                        
                    if len(rels) > 0:
                        output_rels.writerow(rels[0])  # enter rows of relations

class OptimizedCatalog(Module):
    '''
    Populate Taxa into the database in an optimized manor
    '''
    def __init__(self, import_dir=IMPORT, in_label=None, out_label='CatalogFinishedSignal', connect_label=None, name='OptimizedCatalog'):
        Module.__init__(self, in_label, out_label, connect_label, name)
        self.import_dir = import_dir

    def process(self):
        '''
        Exploit LOAD CSV in neo4j to load a taxon catalog efficiently

        '''

        if not os.path.isdir(IMPORT):
            raise RuntimeError('The directory ' + IMPORT + ' was not found. Please update the IMPORT variable in /modules/mining/OptimizedCatalog')

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
