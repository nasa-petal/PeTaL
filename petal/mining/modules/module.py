
class Module:
    def __init__(self, in_label=None, out_label=None, connect_labels=None, instances=1):
        self.in_label      = in_label
        self.out_label     = out_label
        self.connect_labels = connect_labels
        self.instances = instances

    def process(self, node):
        pass
