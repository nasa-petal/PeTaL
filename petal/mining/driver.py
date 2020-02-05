from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json

from modules import WikipediaModule, BackboneModule, EOLModule, GoogleScholarModule, HighwireModule, JEBModule
from utils.neo import page, add_json_node
from uuid import uuid4

# TODO add scheduling etc
class Driver():
    def __init__(self, page_size=100, rate_limit=0.25):
        # self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))
        self.neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))
        self.page_size  = page_size
        self.rate_limit = rate_limit
        self.processes  = dict()

    def paging(self, tx, module):
        finder = 'MATCH (n:{}) '.format(module.in_label)
        query  = finder + 'RETURN n'
        for page_results in page(tx, finder, query, page_size=self.page_size, rate_limit=self.rate_limit):
            for record in page_results.records():
                print(record)
                with self.neo_client.session() as session:
                    node = record['n']
                    result = module.process(node)
                    session.write_transaction(self.write, node, result, module)

    def write(self, tx, node, process_result, module):
        if not isinstance(process_result, list):
            process_result = [process_result]
        for result in process_result:
            if not isinstance(result, tuple):
                in_label  = module.in_label
                out_label = module.out_label
                connect_labels = module.connect_labels
                result = result
            else:
                in_label, out_label, connect_labels, result = result
            id1 = node['uuid']
            id2 = self.add(result, out_label)
            if connect_labels is not None:
                self.link(tx, id1, id2, in_label, out_label, *connect_labels)

    def link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\' MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)'.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label))
        tx.run(query)

    def add(self, data, label):
        unique_id = uuid4()
        data['uuid'] = str(unique_id)
        with self.neo_client.session() as session:
            session.write_transaction(add_json_node, label, data)
        return data['uuid']

    def run(self, module):
        if module.in_label is None:
            for node_json in module.process():
                self.add(node_json, module.out_label)
        else:
            with self.neo_client.session() as session:
                session.read_transaction(self.paging, module)

if __name__ == '__main__':
    driver = Driver(page_size=1, rate_limit=0.25)
    wiki_scraper = WikipediaModule()
    eol_scraper = EOLModule()
    scholar_scraper = GoogleScholarModule()
    backbone = BackboneModule()
    highwire = HighwireModule()
    jeb      = JEBModule()
    # driver.run(backbone)
    # driver.run(wiki_scraper)
    # driver.run(eol_scraper)
    # driver.run(scholar_scraper)
    # driver.run(highwire)
    driver.run(jeb)
