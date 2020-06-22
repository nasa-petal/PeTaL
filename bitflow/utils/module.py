from .transaction import Transaction
from .log import Log
from .profile import Profile

'''
The base class for data bitflow Modules
'''

class Module:
    def __init__(self, in_label=None, out_label=None, connect_labels=None, name='Default', page_batches=False, save=True):
        self.name           = name
        self.in_label       = in_label
        self.out_label      = out_label
        self.connect_labels = connect_labels
        self.page_batches   = page_batches
        self.log            = Log(name, directory='modules')
        self.driver = None
        self.save_batch = save

    def add_driver(self, driver):
        self.driver = driver

    def __enter__(self):
        self.profile = Profile(self.name, directory='modules')
        return self

    def __exit__(self, *args):
        self.profile.close()

    def default_transaction(self, data, uuid=None, from_uuid=None):
        return Transaction(in_label=self.in_label, out_label=self.out_label, connect_labels=self.connect_labels, data=data, uuid=uuid, from_uuid=from_uuid, save=self.save_batch)
    
    def query_transaction(self, query):
        return Transaction(query=query)

    def custom_transaction(self, *args, **kwargs):
        return Transaction(*args, **kwargs)

    def process(self, node):
        raise NotImplementedError()

    def process_batch(self, batch):
        for item in batch.items:
            results = self.process(item)
            if results is None:
                return
            for transaction in results:
                yield transaction

    def __str__(self):
        if self.in_label is None:
            return '{}: ({})'.format(self.name, self.out_label)
        else:
            return '{}: ({}) -> ({})'.format(self.name, self.in_label, self.out_label)
