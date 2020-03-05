from neo4j import GraphDatabase, basic_auth
from pprint import pprint
import json
import neobolt

from utils.neo import page, add_json_node
from collections import defaultdict
from time import sleep, time
from queue import Empty

from batch import Batch

class Driver():
    '''
    An API providing a lightweight connection to neo4j
    '''
    def __init__(self,):
        # self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"), encrypted=False)
        self.neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))
        # self.neo_client = GraphDatabase.driver("bolt://localhost:6969", auth=basic_auth("neo4j", "life"))
        self.hset = set()
        self.lset = set()

    def run(self, transaction):
        if transaction.query is not None:
            with self.neo_client.session() as session:
                session.run(transaction.query)
        else:
            id1 = transaction.from_uuid
            if transaction.data is None:
                id2 = transaction.uuid
            else:
                id2 = self.add(transaction.data, transaction.out_label)
                if id2 in self.hset:
                    return False
                self.hset.add(id2)
            print(id1, flush=True)
            if id1 is not None and transaction.connect_labels is not None:
                id1 = str(id1)
                key = str(id1) + str(id2)
                if key in self.lset:
                    return False
                with self.neo_client.session() as session:
                    self.lset.add(key)
                    session.write_transaction(self.link, id1, id2, transaction.in_label, transaction.out_label, *transaction.connect_labels)
        return True

    def link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\' MERGE (n)-[:{from_label}]->(m) MERGE (m)-[:{to_label}]->(n)'.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2, from_label=from_label, to_label=to_label))
        print(query, flush=True)
        tx.run(query)

    def add(self, data, label):
        with self.neo_client.session() as session:
            node = session.write_transaction(add_json_node, label, data)
            records = node.records()
            node = (next(records)['n'])
            return node['uuid']

    def get(self, uuid):
        with self.neo_client.session() as session:
            records = list(session.run('MATCH (n) WHERE n.uuid = \'{uuid}\' RETURN n'.format(uuid=str(uuid))).records())
        if len(records) > 0:
            return records[0]
        else:
            raise ValueError('UUID {} invalid'.format(uuid))

    def count(self, label):
        with self.neo_client.session() as session:
            records = session.run('MATCH (x:{label}) WITH COUNT (x) AS count RETURN count'.format(label=label)).records()
        return list(records)[0]['count']

def driver_listener(transaction_queue):
    start = time()
    driver = Driver()
    i = 0
    while True:
        batch_file = transaction_queue.get()
        batch = Batch()
        batch.load(batch_file)
        for transaction in batch.items:
            try:
                added = driver.run(transaction)
                duration = time() - start
                total = len(driver.hset) + len(driver.lset)
                print('Driver rate: {} of {} ({}|{})\r'.format(round(total / duration, 3), total, len(driver.hset), len(driver.lset)), flush=True, end='')
                if added:
                    i += 1
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except neobolt.exceptions.CypherSyntaxError:
                pass
            except Exception as e:
                print(e, flush=True)
                print(transaction.in_label, flush=True)
                print(transaction.out_label, flush=True)
                print(transaction.uuid, flush=True)
                print(transaction.from_uuid, flush=True)
                print(transaction.data, flush=True)

