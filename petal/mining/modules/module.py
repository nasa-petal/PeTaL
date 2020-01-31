
class Module:
    def __init__(self, in_label=None, out_label=None, connect_label=None):
        self.in_label      = in_label
        self.out_label     = out_label
        self.connect_label = connect_label

    def process(self, node):
        pass
