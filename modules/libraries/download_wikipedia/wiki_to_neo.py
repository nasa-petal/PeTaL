from pprint import pprint
from subprocess import call
from time import time, sleep

from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-b91ac8a8-e7bf-46a0-9651-1e7b068e5919/installation-3.5.14/import/'
CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-96095927-f047-445d-8ce8-b4b05024bc48/installation-3.5.14/import/'

def main():
    # print('Copying files..', flush=True)
    # for filename in os.listdir('.'):
    #     if filename.endswith('.csv'):
    #         shutil.copy(filename, CUSTOM_IMPORT_DIR + filename)
    print('Beginning entries', flush=True)
    neo_client = GraphDatabase.driver("bolt://localhost:6969", auth=basic_auth("neo4j", "life"))
    with neo_client.session() as session:
        headers = ['title', 'redirect']
        # session.run('CREATE INDEX ON :WikipediaPage(title)')
        # session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///pages.csv" AS line CREATE (x:WikipediaPage {' + ','.join(h + ': line.' + h for h in headers) + '})') 
        session.run('USING PERIODIC COMMIT 500 LOAD CSV WITH HEADERS FROM "file:///links.csv" AS line MATCH (x:WikipediaPage {title: line.from}),(y:WikipediaPage {title: line.to}) CREATE (x)-[:link]->(y)') # (y)-[:subtaxon]->(x)')
        # session.run('USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM "file:///links.csv" AS line MATCH (x:WikipediaPage {name: line.from}),(y:WikipediaPage {name: line.to}) CREATE (x)-[:link]->(y)') # (y)-[:subtaxon]->(x)')

    print('Done!', flush=True)

if __name__ == '__main__':
    main()
