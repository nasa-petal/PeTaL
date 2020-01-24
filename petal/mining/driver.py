from eol  import search as eol_search
from wiki import search as wiki_search
from scholarly import search_pubs_query as google_scholar_search

from neo4j import GraphDatabase

neo_uri = "bolt://localhost:7687"
neo_client = GraphDatabase.driver(neo_uri, auth=("neo4j", "life"))

import json

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
    scholar_results = google_scholar_search(name + ' source:"journal of experimental biology"')
    print(next(scholar_results))
    return
    # results = eol_search(name)

def main():
    with neo_client.session() as session:
        session.read_transaction(count_species)
        session.read_transaction(iter_species, mapper)

if __name__ == '__main__':
    main()
