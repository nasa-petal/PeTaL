from uuid import uuid4

class Transaction:
    def __init__(self, in_label=None, out_label=None, connect_labels=None, data=None, query=None, uuid=None):
        if uuid is None:
            uuid = str(uuid4())
        self.in_label       = in_label
        self.out_label      = out_label
        self.connect_labels = connect_labels
        self.data           = data
        self.query          = query
        self.uuid           = uuid
