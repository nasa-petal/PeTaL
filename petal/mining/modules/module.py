
class Module:
    def __init__(self, in_label=None, out_label=None, connect_labels=None):
        self.in_label      = in_label
        self.out_label     = out_label
        self.connect_labels = connect_labels

    def process(self, node):
        pass
