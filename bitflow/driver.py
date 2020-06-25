from neo4j import GraphDatabase, basic_auth
import json
import neobolt

from collections import defaultdict
from time import sleep, time
from queue import Empty

from .batch import Batch

from .utils.log import Log
from .utils.profile import Profile
from .utils.transaction import Transaction
from .utils.utils import add_json_node
from .utils.utils import clean_uuid as clean

'''
This file defines a wrapper around the neo4j database. This code is pretty messy, as it has evolved a lot over time.
'''

def retry(f):
    '''
    Decorator. Quality of life: keeps certain functions going even if the database server goes down for some reason
    Usage:
    @retry
    def my_function(arg):
        # do something important with the database connection
    '''
    def inner(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except neobolt.exceptions.ServiceUnavailable as e:
                print('Cannot reach neo4j server. Is it running? Sleeping 1s..', flush=True)
                sleep(1)
    return inner

class Driver():
    '''
    An API providing a lightweight connection to neo4j
    Note: I wrote "lightweight" when that was true. Not clear if still true. Hopefully stays close to true.
    '''
    @retry
    def __init__(self, settings_file):
        '''
        :param settings_file: Path to a json file containing neo4j server settings
        '''
        with open(settings_file, 'r') as infile:
            settings = json.load(infile)
        self.neo_client = GraphDatabase.driver(settings["neo4j_server"], auth=basic_auth(settings["username"], settings["password"]), encrypted=settings["encrypted"])
        self.session = self.neo_client.session()
        self.hset = set()
        self.lset = set()

    @retry
    def run_query(self, query):
        '''
        :param query: A string containing a neo4j query
        '''
        return self.session.run(query)

    @retry
    def run(self, transaction):
        '''
        :param transaction: A :class:`utils.transaction.Transaction` representing an operation on the database
        '''
        if transaction.query is not None:
            self.session.run(transaction.query)
        else:
            id1 = clean(transaction.from_uuid)
            id2 = clean(transaction.uuid)
            # Add a new node to neo4j
            if transaction.data is not None:
                if id2 in self.hset:
                    return False
                self.hset.add(id2)
                self.session.write_transaction(add_json_node, transaction.out_label, transaction.data)
            # Add a new connection to neo4j
            if id1 is not None and transaction.connect_labels is not None:
                id1 = str(id1)
                key = str(id1) + str(id2)
                if key in self.lset:
                    return False
                self.lset.add(key)
                self.session.write_transaction(self._link, id1, id2, transaction.in_label, transaction.out_label, *transaction.connect_labels)
            return True

    def _link(self, tx, id1, id2, in_label, out_label, from_label, to_label):
        '''
        Link two nodes in neo4j, with the given ids and labels
        '''
        from_label = clean(from_label).replace(' ', '_')
        to_label   = clean(to_label).replace(' ', '_')
        query = ('MATCH (n:{in_label}) WHERE n.uuid=\'{id1}\' MATCH (m:{out_label}) WHERE m.uuid=\'{id2}\''.format(in_label=in_label, out_label=out_label, id1=id1, id2=id2))
        if from_label is not None:
            query += (' MERGE (n)-[:{from_label}]->(m)'.format(from_label=from_label))
        if to_label is not None:
            query += (' MERGE (n)-[:{to_label}]->(m)'.format(to_label=to_label))
        tx.run(query)

    @retry
    def get(self, uuid):
        '''
        Return the node pointed to by uuid. Returns None if not present
        :param uuid: The uuid of a neo4j node
        '''
        uuid = clean(uuid)
        records = list(self.session.run('MATCH (n) WHERE n.uuid = \'{uuid}\' RETURN n'.format(uuid=str(uuid))).records())
        if len(records) > 0:
            return records[0]['n']
        else:
            return None

    @retry
    def count(self, label):
        '''
        Count the number of nodes with a particular label in the database
        :param label: The label to count nodes of
        '''
        records = self.session.run('MATCH (x:{label}) WITH COUNT (x) AS count RETURN count'.format(label=label)).records()
        return list(records)[0]['count']

def driver_listener(transaction_queue, settings_file):
    '''
    The continually running driver process which allows other data-generating processes to run while transactions are put in line
    :param transaction_queue: A :class:`multiprocessing.Queue` object full of :class:`Batch` objects.
    :param settings_file: A json file with Driver-specific settings, like address and password... maybe someone should change this?
    '''
    profile = Profile('driver')
    log     = Log('driver')

    start = time()
    driver = Driver(settings_file)
    while True:
        # Wait for batches and then add them to the database
        batch = transaction_queue.get()
        for transaction in batch.items:
            log.log(transaction)
            try:
                driver.run(transaction)
            except TypeError as e:
                log.log(e)
        # Save batches to database for re-use
        if batch.save and batch.label is not None:
            for sublabel in batch.label.split(':'):
                driver.run(Transaction(out_label='Batch', data={'label' : sublabel, 'filename' : batch.filename, 'rand' : batch.rand}, uuid=batch.uuid))
        duration = time() - start
        total = len(driver.hset) + len(driver.lset)
        log.log('Driver rate: {} of {} ({}|{})'.format(round(total / duration, 3), total, len(driver.hset), len(driver.lset)))
        log.log('Created batch for ', batch.label)
