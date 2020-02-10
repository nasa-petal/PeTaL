from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json

from utils.neo import page, add_json_node
from uuid import uuid4
from collections import defaultdict

from multiprocessing import Lock

from copy import deepcopy

# TODO add scheduling etc
class Driver():
    def __init__(self, page_size=1, rate_limit=0.25):
        # self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))
        self.neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))
        self.page_size  = page_size
        self.rate_limit = rate_limit
        self.tracker = None

    def write(self, tx, node, process_result, module):
        # TODO: Simplify this interface? Currently allows [dict], [str], [tuple], dict, str, tuple from module.process()
        if not isinstance(process_result, list):
            process_result = [process_result]
        for result in process_result:
            if isinstance(result, str): # Result is a query
                tx.run(result)
            else:
                if isinstance(result, dict):
                    in_label  = module.in_label
                    out_label = module.out_label
                    connect_labels = module.connect_labels
                    result = result
                elif isinstance(result, tuple):
                    in_label, out_label, connect_labels, result = result
                else:
                    raise ValueError('Invalid scraping module process(node) result: {}'.format(result))
                id1 = node['uuid']
                id2 = self.add(result, out_label)
                if connect_labels is not None:
                    self.link(tx, id1, id2, in_label, out_label, *connect_labels)

    def link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\' MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)'.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label))
        tx.run(query)

    def add(self, data, label):
        with self.neo_client.session() as session:
            node = session.write_transaction(add_json_node, label, data)
            records = node.records()
            node = (next(records)['n'])
            id_n = node.id
            if 'uuid' not in node:
                unique_id = uuid4()
                session.run('MATCH (s) WHERE ID(s) = {} SET s.uuid = \'{}\' RETURN s'.format(id_n, str(unique_id)))
            else:
                unique_id = node['uuid']
            if self.tracker is not None:
                self.tracker.add(label, unique_id)
            return unique_id

    def page_runner(self, tx, module, ids):
        for node_id in ids:
            node = tx.run('MATCH (n) WHERE n.uuid = \'' + node_id + '\' RETURN n')
            for record in node.records():
                with self.neo_client.session() as session:
                    node = record['n']
                    print(node, flush=True)
                    result = module.process(node)
                    session.write_transaction(self.write, node, result, module)

    def run_page(self, module, tracker, ids):
        self.tracker = tracker
        with self.neo_client.session() as session:
            session.read_transaction(self.page_runner, module, ids)

    def run(self, module, tracker):
        self.tracker = tracker
        for node_json in module.process():
            self.add(node_json, module.out_label)
