from ..utils.module import Module

class DependentTestModule(Module):
    def __init__(self, in_label='TestOutput', out_label='DependentTestOutput', connect_label=None, name='DependentTester', count=1):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self, previous):
        name = previous.data['name'] + '!'
        yield self.default_transaction({'name' : name}, uuid=name, from_uuid=previous.uuid)
