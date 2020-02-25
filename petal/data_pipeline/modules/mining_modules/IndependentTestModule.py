from ..utils.module import Module

class IndependentTestModule(Module):
    def __init__(self, in_label=None, out_label='TestOutput', connect_label=None, name='IndependentTester', count=1):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self):
        for name in 'abcdefghijklmnopqrstuvwxyz':
            yield self.default_transaction({'name' : name}, uuid=name)
