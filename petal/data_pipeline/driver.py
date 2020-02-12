from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json

from utils.neo import page, add_json_node
from uuid import uuid4
from collections import defaultdict
from time import sleep

class Driver():
    '''
    An API providing a lightweight connection to neo4j
    '''
    def __init__(self):
        # self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7667", auth=basic_auth("neo4j", "testing"))
        self.neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))
        self.tracker = None

    def write(self, tx, node, transactions, module):
        for transaction in transactions:
            if transaction is None:
                pass
            if transaction.query is not None:
                tx.run(transaction.query)
            else:
                id1 = node['uuid']
                id2 = self.add(transaction.data, transaction.out_label)
                if transaction.connect_labels is not None:
                    self.link(tx, id1, id2, transaction.in_label, transaction.out_label, *transaction.connect_labels)

    def link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\' MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)'.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label))
        tx.run(query)

    def add(self, data, label):
        with self.neo_client.session() as session:
            while self.tracker.count(label) > self.tracker.throttle_count(label):
                sleep(0.2)
                # print('Sleeping until products are consumed', flush=True)
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

    def page_runner(self, tx, module, node_id):
        node = tx.run('MATCH (n) WHERE n.uuid = \'' + node_id + '\' RETURN n')
        for record in node.records():
            with self.neo_client.session() as session:
                node = record['n']
                transactions = module.process(node)
                session.write_transaction(self.write, node, transactions, module)

    def run_id(self, module, tracker, node_id):
        self.tracker = tracker
        with self.neo_client.session() as session:
            session.read_transaction(self.page_runner, module, node_id)

    def run(self, module, tracker, info):
        self.tracker = tracker
        i = 0
        for transaction in module.process():
            info.set_current(i)
            self.add(transaction.data, transaction.out_label)
            i += 1
