from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json

from modules import WikipediaModule, BackboneModule # Automatically import?
from utils.neo import page, add_json_node

class Driver():
    def __init__(self, page_size=100, rate_limit=0.25):
        self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))
        self.page_size  = page_size
        self.rate_limit = rate_limit
        self.processes  = dict()

    def paging(self, tx, module):
        finder = 'MATCH (n:{}) '.format(module.in_label)
        query  = finder + 'RETURN n'
        for page_results in page(tx, finder, query, page_size=self.page_size, rate_limit=self.rate_limit):
            for record in page_results.records():
                with self.neo_client.session() as session:
                    result = module.process(record['n'])
                    self.write(result, module)

    def write(self, process_result, module):
        if len(process_result) > 0:
            pprint(process_result) # TODO add to db here
            if module.connect_label is None:
                pass
            else:
                print('Connect nodes here') # TODO add to db here
        print('.', end='', flush=True)

    def run(self, module):
        if module.in_label is None:
            for node_json in module.process():
                with self.neo_client.session() as session:
                    session.write_transaction(add_json_node, module.out_label, node_json)
        else:
            with self.neo_client.session() as session:
                session.read_transaction(self.paging, module)

if __name__ == '__main__':
    driver = Driver(page_size=10, rate_limit=0.25)
    wiki_scraper = WikipediaModule()
    backbone = BackboneModule()
    driver.run(backbone)
    driver.run(wiki_scraper)
