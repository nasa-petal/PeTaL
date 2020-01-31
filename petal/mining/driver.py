from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json

from modules import WikipediaModule, BackboneModule # Automatically import?
from utils.neo import page, add_json_node
from uuid import uuid4

# TODO add scheduling etc


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
                    node = record['n']
                    result = module.process(node)
                    session.write_transaction(self.write, node, result, module)

    def write(self, tx, node, process_result, module):
        if len(process_result) > 0:
            pprint(process_result) # TODO add to db here
            if module.connect_label is None:
                pass
            else:
                print('Connect nodes here') # TODO add to db here
        print('.', end='', flush=True)
        id1 = node['uuid']
        id2 = self.add(process_result, module.out_label)
        if module.connect_labels is not None:
            self.link(tx, id1, id2, module)

    def link(self, tx, id1, id2, module):
        from_label, to_label = module.connect_labels
        print('MATCH (n:{in_label}) WHERE n.uuid={id1} MATCH (m:{out_label}) WHERE m.uuid={id2} MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)'.format(in_label=module.in_label, out_label=module.out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label))
        1/0
        # tx.run('MATCH (n: {in_label}) WHERE n.uuid={id1} MATCH (m:out_label) WHERE m.uuid={id2} MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)', in_label=module.in_label, out_label=module.out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label)

    def add(self, data, label):
        unique_id = uuid4()
        data['uuid'] = str(unique_id)
        with self.neo_client.session() as session:
            session.write_transaction(add_json_node, label, data)
        return unique_id

    def run(self, module):
        if module.in_label is None:
            for node_json in module.process():
                self.add(node_json, module.out_label)
        else:
            with self.neo_client.session() as session:
                session.read_transaction(self.paging, module)

if __name__ == '__main__':
    driver = Driver(page_size=10, rate_limit=0.25)
    wiki_scraper = WikipediaModule()
    # backbone = BackboneModule()
    # driver.run(backbone)
    driver.run(wiki_scraper)
