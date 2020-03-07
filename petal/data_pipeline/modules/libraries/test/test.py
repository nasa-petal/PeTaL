from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-b91ac8a8-e7bf-46a0-9651-1e7b068e5919/installation-3.5.14/import/'
CUSTOM_IMPORT_DIR = '../../../../../../../.Neo4jDesktop/neo4jDatabases/database-96095927-f047-445d-8ce8-b4b05024bc48/installation-3.5.14/import/'

def main():
    for filename in os.listdir('.'):
        if filename.endswith('.csv'):
            shutil.copy(filename, CUSTOM_IMPORT_DIR + filename)
    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))

    with neo_client.session() as session:
         session.run('CREATE CONSTRAINT ON (person:Person) ASSERT person.id IS UNIQUE')
         session.run('CREATE CONSTRAINT ON (movie:Movie) ASSERT movie.id IS UNIQUE')
         session.run('CREATE INDEX ON :Country(name)')
         session.run('LOAD CSV WITH HEADERS FROM "file:///persons.csv" AS csvLine CREATE (p:Person {id: toInteger(csvLine.id), name: csvLine.name})')
         session.run('LOAD CSV WITH HEADERS FROM "file:///movies.csv" AS csvLine MERGE (country:Country {name: csvLine.country}) CREATE (movie:Movie {id: toInteger(csvLine.id), title: csvLine.title, year:toInteger(csvLine.year)}) CREATE (movie)-[:MADE_IN]->(country)')
         session.run('USING PERIODIC COMMIT 500 LOAD CSV WITH HEADERS FROM "file:///roles.csv" AS csvLine MATCH (person:Person {id: toInteger(csvLine.personId)}),(movie:Movie {id: toInteger(csvLine.movieId)}) CREATE (person)-[:PLAYED {role: csvLine.role}]->(movie)')
    print('Done!', flush=True)

if __name__ == '__main__':
    main()
