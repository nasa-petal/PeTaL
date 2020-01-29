from eol  import search as eol_search
from wiki import search as wiki_search
from scholarly import search_pubs_query as google_scholar_search

from neo4j import GraphDatabase, basic_auth
from neo import add_json, page

# neo_uri = "bolt://localhost:7687"
# neo_client = GraphDatabase.driver(neo_uri, auth=("neo4j", "life"))
neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))

GOOGLE_SCHOLAR_ARTICLE_LIMIT = 10

import json

def iter_species(tx, mapper):
    species_result = tx.run('MATCH (n:Species) RETURN n LIMIT 10')
    species = species_result.records()
    for s in species:
        mapper(s['n'], tx)

def count_species(tx):
    result = tx.run('MATCH (n) WITH COUNT (n) AS count RETURN count')
    saved = result.single()
    print('neo4j currently holds ', saved['count'], ' species')
    
def add_connection(tx, species, article):
    title = article.bib['title']
    tx.run('MATCH (n:Article) WHERE n.title={title} MATCH (m:Species) WHERE m.Name={name} MERGE (n)-[:MENTIONS_SPECIES]->(m) MERGE (m)-[:MENTIONED_IN_ARTICLE]->(n)', title=title, name=species['Name'])

def add_species_article(tx, article, species):
    base = article.bib
    add_json(tx, 'Article', base)
    add_connection(tx, species, article)


def mapper(species, tx):
    name = species['Name']
    print('Mapper on species: ', name)
    scholar_results = google_scholar_search(name)
    for i, article in enumerate(scholar_results):
        add_species_article(tx, article, species)
        if i == GOOGLE_SCHOLAR_ARTICLE_LIMIT:
            break
    return
    # results = eol_search(name)

def tester(tx):
    for item in page(tx, 'MATCH (n:Species)', 'MATCH (n:Article) RETURN n'):
        print(item)

def main():
    with neo_client.session() as session:
        session.read_transaction(tester)
        # session.read_transaction(count_species)
        # session.read_transaction(iter_species, mapper)

if __name__ == '__main__':
    main()
