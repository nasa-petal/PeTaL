from pprint import pprint
from subprocess import call
from time import time

import requests, zipfile, os

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

def to_json():
    '''
    All that this function does is yield Transaction() objects which create Species() nodes in the neo4j database.
    This particular process() function is simply downloading a tab-separated file and parsing it.
    '''
    seen = set()
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
            elif i % 10000 == 0:
                print(i, flush=True)
            elif i == 100000:
                break
            else:
                for k, v in zip(headers, line.split('\t')):
                    json[k] = v
                try:
                    json.pop('isExtinct\n')
                except KeyError:
                    pass
                if json['taxonRank'] == 'species':
                    json['name'] = json['scientificName'].replace(json['scientificNameAuthorship'], '').strip()
                    yield json
                    # yield self.default_transaction(json, uuid=json['name']) # HERE is where the transaction is created!!
                    # last_uuid = json['name']
                    # last_label = 'Species:Taxon'
                    # for taxon in ['subgenus', 'genus', 'family', 'superfamily', 'order', 'class', 'phylum', 'kingdom']:
                    #     name = json[taxon]
                    #     if name.strip() == '':
                    #         continue
                    #     if (last_uuid, name) not in seen:
                    #         data = dict(name=name, uuid=name)
                    #         label_name = taxon[0].upper() + taxon[1:] + ':Taxon'
                    #         seen.add((last_uuid, name))
                    #         yield self.custom_transaction(data=data, in_label=last_label, out_label=label_name, connect_labels=('supertaxon', 'subtaxon'), uuid=name, from_uuid=last_uuid)
                    #     last_label = label_name
                    #     last_uuid = name
                json = dict()
            i += 1

def to_csv():
    first = True
    with open('catalog.csv', 'w', encoding='utf-8') as outfile:
        for entry in to_json():
            if first:
                outfile.write(','.join(entry.keys()) + '\n')
                first = False
            outfile.write(','.join(entry.values()) + '\n')

from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-b91ac8a8-e7bf-46a0-9651-1e7b068e5919/installation-3.5.14/import/'
CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-96095927-f047-445d-8ce8-b4b05024bc48/installation-3.5.14/import/'

def main():
    to_csv()
    for filename in os.listdir('.'):
        if filename.endswith('.csv'):
            shutil.copy(filename, CUSTOM_IMPORT_DIR + filename)
    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))


    with neo_client.session() as session:
         headers = 'id,identifier,datasetID,datasetName,acceptedNameUsageID,parentNameUsageID,taxonomicStatus,taxonRank,verbatimTaxonRank,scientificName,kingdom,phylum,class,order,superfamily,family,genericName,genus,subgenus,specificEpithet,infraspecificEpithet,scientificNameAuthorship,source,namePublishedIn,nameAccordingTo,modified,description,taxonConceptID,scientificNameID,references,name'.split(',')
         session.run('CREATE INDEX ON :Taxon(name)')
         session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///catalog.csv" AS line CREATE (x:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')

         # session.run('CREATE CONSTRAINT ON (person:Person) ASSERT person.id IS UNIQUE')
         # session.run('CREATE CONSTRAINT ON (movie:Movie) ASSERT movie.id IS UNIQUE')
         # session.run('LOAD CSV WITH HEADERS FROM "file:///movies.csv" AS csvLine MERGE (country:Country {name: csvLine.country}) CREATE (movie:Movie {id: toInteger(csvLine.id), title: csvLine.title, year:toInteger(csvLine.year)}) CREATE (movie)-[:MADE_IN]->(country)')
         # session.run('USING PERIODIC COMMIT 500 LOAD CSV WITH HEADERS FROM "file:///roles.csv" AS csvLine MATCH (person:Person {id: toInteger(csvLine.personId)}),(movie:Movie {id: toInteger(csvLine.movieId)}) CREATE (person)-[:PLAYED {role: csvLine.role}]->(movie)')

if __name__ == '__main__':
    main()
