from eol  import search as eol_search
from wiki import search as wiki_search

from neo4j import GraphDatabase

neo_uri = "bolt://localhost:7687"
neo_client = GraphDatabase.driver(neo_uri, auth=("neo4j", "life"))

from pymongo import MongoClient
import json

# Connecting to the database
mongo_db_name = 'test_db'
username = 'test_user'
password = 'testing'
mongo_client = MongoClient('mongodb://{0}:{1}@139.88.179.199:27017/{2}'.format(username,password, mongo_db_name),maxPoolSize=50)


def iter_species(tx, mapper):
    species_result = tx.run('MATCH (n:Species) RETURN n')
    species = species_result.records()
    for s in species:
        mapper(s['n'])
        break

def count_species(tx):
    result = tx.run('MATCH (n) WITH COUNT (n) AS count RETURN count')
    saved = result.single()
    print('neo4j currently holds ', saved['count'], ' species')

def mapper(species):
    name = species['Name']
    print('Mapper on species: ', name)
    results = eol_search(name)
    print(results)
    with mongo_client:
        eol_db = mongo_client.eol_db
        eol_db.eol_db.insert_one({name : results})

def main():
    with neo_client.session() as session:
        session.read_transaction(count_species)
        session.read_transaction(iter_species, mapper)

if __name__ == '__main__':
    main()
