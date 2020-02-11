from .transaction import Transaction

class Module:
    def __init__(self, in_label=None, out_label=None, connect_labels=None, name='Default', count=1):
        self.in_label      = in_label
        self.out_label     = out_label
        self.connect_labels = connect_labels
        self.name = name
        self.count = count

    def default_transaction(self, data):
        return Transaction(in_label=self.in_label, out_label=self.out_label, connect_labels=self.connect_labels, data=data)
    
    def query_transaction(self, query):
        return Transaction(query=query)

    def custom_transaction(self, *args, **kwargs):
        return Transaction(*args, **kwargs)

    def process(self, node):
        pass

    def __str__(self):
        if self.in_label is None:
            return '{}: ({})'.format(self.name, self.out_label)
        else:
            return '{}: ({}) -> ({})'.format(self.name, self.in_label, self.out_label)
        # if self.connect_labels is not None:
        #     return '{}: {} <-[{}, {}]-> {}'.format(self.name, self.in_label, *self.connect_labels, self.out_label)

