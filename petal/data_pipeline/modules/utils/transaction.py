# from uuid import uuid4
import hashlib

class Transaction:
    def __init__(self, in_label=None, out_label=None, connect_labels=None, data=None, query=None, uuid=None, from_uuid=None):
        if uuid is None:
            if data is not None:
                print('Had to generate new UUID', flush=True)
                uuid = str(hashlib.md5(','.join(map(str, sorted(data.items())))).encode('utf-8').hexdigest())
        self.in_label       = in_label
        self.out_label      = out_label
        self.connect_labels = connect_labels
        self.data           = data
        self.query          = query
        self.uuid           = uuid
        self.from_uuid      = from_uuid
        if self.data is not None:
            self.data['uuid'] = str(self.uuid)
