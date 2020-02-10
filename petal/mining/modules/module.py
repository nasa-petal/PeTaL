
class Module:
    def __init__(self, in_label=None, out_label=None, connect_labels=None, name='Default'):
        self.in_label      = in_label
        self.out_label     = out_label
        self.connect_labels = connect_labels
        self.name = name

    def process(self, node):
        pass

    def __str__(self):
        if self.in_label is None:
            return '{}: ({})'.format(self.name, self.out_label)
        else:
            return '{}: ({}) -> ({})'.format(self.name, self.in_label, self.out_label)
        # if self.connect_labels is not None:
        #     return '{}: {} <-[{}, {}]-> {}'.format(self.name, self.in_label, *self.connect_labels, self.out_label)

