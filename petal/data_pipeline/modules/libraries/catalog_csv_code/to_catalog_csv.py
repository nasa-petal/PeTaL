from pprint import pprint
from subprocess import call
from time import time, sleep

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
                # break
            else:
                for k, v in zip(headers, line.split('\t')):
                    json[k] = v.replace('"', '')
                try:
                    json.pop('isExtinct\n')
                except KeyError:
                    pass
                rank = json['taxonRank']
                if rank == 'species' or rank == 'infraspecies':
                    json['name'] = json['scientificName'].replace(json['scientificNameAuthorship'], '').strip()
                else:
                    json['name'] = json[rank]
                if json['name'] == 'Not assigned':
                    continue
                found = False
                relations = []
                for taxon in ['species', 'subgenus', 'genus', 'family', 'superfamily', 'order', 'class', 'phylum', 'kingdom']:
                    if found:
                        if json[taxon].strip() != '':
                            relations.append((json['name'], json[taxon]))
                            break
                    if rank == 'infraspecies':
                        if taxon == 'species': # Necessary
                            found = True
                    elif taxon == rank:
                        found = True
                yield json, relations
                # last_uuid = json['name']
                #     json['taxonRank'] = taxon
                #     name = json[taxon]
                #     if name.strip() == '':
                #         continue
                #     json['name'] = name
                #     yield json, [(last_uuid, name)]
                #     last_uuid = name
                #     json[taxon] = ''
                json = dict()
            i += 1

def to_csv():
    first = True
    with open('catalog.csv', 'w', encoding='utf-8') as catalog:
        with open('species.csv', 'w', encoding='utf-8') as species_csv:
            with open('relations.csv', 'w', encoding='utf-8') as relations:
                for entry, rels in to_json():
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

from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-b91ac8a8-e7bf-46a0-9651-1e7b068e5919/installation-3.5.14/import/'
CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-96095927-f047-445d-8ce8-b4b05024bc48/installation-3.5.14/import/'

def main():
    if not os.path.isfile('catalog.csv') or not os.path.isfile('relations.csv'):
        to_csv()
    for filename in os.listdir('.'):
        if filename.endswith('.csv'):
            shutil.copy(filename, CUSTOM_IMPORT_DIR + filename)
    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))


    with neo_client.session() as session:
         headers = 'id,identifier,datasetID,datasetName,acceptedNameUsageID,parentNameUsageID,taxonomicStatus,taxonRank,verbatimTaxonRank,scientificName,kingdom,phylum,class,order,superfamily,family,genericName,genus,subgenus,specificEpithet,infraspecificEpithet,scientificNameAuthorship,source,namePublishedIn,nameAccordingTo,modified,description,taxonConceptID,scientificNameID,references,name'.split(',')
         session.run('CREATE INDEX ON :Taxon(name)')
         # session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///catalog.csv" AS line CREATE (x:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
         # session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///species.csv" AS line CREATE (x:Species:Taxon {' + ','.join(h + ': line.' + h for h in headers) + '})')
         session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///relations.csv" AS line MATCH (x:Taxon {name: line.from}),(y:Taxon {name: line.to}) CREATE (x)-[:supertaxon]->(y)') # (y)-[:subtaxon]->(x)')

    print('Done!', flush=True)

if __name__ == '__main__':
    main()
