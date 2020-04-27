from pprint import pprint
from subprocess import call
from time import time, sleep

import requests, zipfile, os
from neo4j import GraphDatabase, basic_auth

from time import sleep
import shutil, os

from petal.pipeline.module_utils.module import Module

NEO4J_IMPORT_DIR = '../../.Neo4jDesktop/neo4jDatabases/database-c60209c1-62b4-4cb3-90be-59a9b62b4141/installation-4.0.2/import/'


class DirectoryParser(Module):
    '''
    Populate neo4j with NASA directory info
    '''
    def __init__(self, import_dir=NEO4J_IMPORT_DIR, in_label=None, out_label='DirectorParserFinishedSignal', connect_label=None, name='Directory Parser'):
        Module.__init__(self, in_label, out_label, connect_label, name)
        self.import_dir = import_dir

    def read(self, filename, headers_index=0):
        with open(filename, 'r') as infile:
            residuals = None
            for i, line in enumerate(infile):
                cells = [cell.strip() for cell in line.split(',')]
                if i == headers_index:
                    yield cells
                elif i > headers_index:
                    if residuals is None:
                        residuals = cells
                    else:
                        residuals = [r if cell.strip() == '' else cell for cell, r in zip(cells, residuals)]
                    yield residuals

    def long_form(self, filename, headers_index=0):
        reader  = self.read(filename, headers_index=headers_index)
        headers = next(reader)
        data    = {h.replace(' ', '_') : [] for h in headers}
        for row in reader:
            for label, cell in zip(data.keys(), row):
                data[label].append(cell)
        return data

    def add_hierarchy(self, long_data):
        previous  = None
        from_uuid = None
        for row in zip(*long_data.values()):
            for key, value in zip(long_data.keys(), row):
                data = dict(name=value)
                uuid = value + '_' + key
                yield self.custom_transaction(data=data, in_label=previous, out_label=key, uuid=uuid, from_uuid=from_uuid, connect_labels=('sub', 'super'))
                previous  = key
                from_uuid = uuid

    def link_people(self, people):
        for row in zip(*people.values()):
            row_dict = {k : v for k, v in zip(people.keys(), row)}
            name    = row_dict['Name']
            project = row_dict['Project_name']
            yield self.custom_transaction(in_label='Name', out_label='Project', from_uuid=name + '_Name', uuid=project + '_Project')

    def process(self):
        # Read CSV files into long form
        directory    = 'data/directory_data/'
        people       = self.long_form(directory + 'people.csv', headers_index=1)
        projects     = self.long_form(directory + 'projects.csv')
        data_science = self.long_form(directory + 'data_science.csv')

        # Rename and remove some columns
        project_descriptions = projects.pop('Description')

        transaction_pool = []
        transaction_pool.append(self.add_hierarchy(data_science))
        transaction_pool.append(self.add_hierarchy(projects))
        transaction_pool.append(self.add_hierarchy(people))
        transaction_pool.append(self.link_people(people))

        for transactions in transaction_pool:
            for transaction in transactions:
                yield transaction




