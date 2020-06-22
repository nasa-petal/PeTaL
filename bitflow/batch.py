import pickle, os
from random import random

from .utils.utils import clean_uuid

class Batch:
    '''
    A data storage class used within the bitflow.
    Literally just a serializable labeled list of items with a piece of random information attached,
    which can be used to separate data into categories (conventionally test, train, validate)
    '''
    def __init__(self, label, uuid=None, rand=None):
        '''
        :param label: A neo4j label
        :param uuid: A unique string denoting the batch, must be given.
        :param rand: A number between 0.0 and 1.0, randomly generated if not given
        '''
        self.items    = []
        self.label    = label
        if uuid is None:
            raise ValueError('Batch was supplied with UUID None')
        self.uuid = clean_uuid(uuid)
        self.filename = 'data/batches/' + str(self.uuid)
        if rand is None:
            self.rand = random()
        else:
            self.rand = rand

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.append(item)

    def save(self):
        '''
        Serialize to uuid-named file in data/batches
        '''
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.items, outfile)

    def load(self):
        '''
        Serialize from uuid-named file in data/batches
        '''
        if os.path.isfile(self.filename):
            with open(self.filename, 'rb') as infile:
                self.items = pickle.load(infile)
        else:
            raise OSError('No batch file ' + self.filename)
