from neo4j import GraphDatabase

import os
import pickle

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "life"))

def add_species_list(tx, species_list):
    for k, p, c, o, f, g, name in species_list:
        properties = {
                'kingdom' : k,
                'phylum'  : p,
                'class'   : c,
                'order'   : o,
                'family'  : f,
                'genus'   : g,
                'name'    : name
                }
        prop_field = '{' + ','.join('{key}: {{{key}}}'.format(key=k) for k in properties) + '}'
        query = 'CREATE (n:Species ' + prop_field + ')'
        tx.run(query, **properties)
                          
def get_species_lists(root):
    for filename in os.listdir(root):
        with open(root + '/' + filename, 'rb') as infile:
            species_list = pickle.load(infile)
            yield species_list

def main():
    with driver.session() as session:
        for species_list in get_species_lists('data'):
            session.read_transaction(add_species_list, species_list)

if __name__ == '__main__':
    main()
