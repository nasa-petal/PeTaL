from petal.pipeline.module_utils.module import Module

class MockMLData(Module):
    def __init__(self, in_label=None, out_label='MLData', connect_labels=None, name='MockMLData'):
        Module.__init__(self, in_label=in_label, out_label=out_label, connect_labels=connect_labels, name=name)

    def process(self, driver=None):
        for i in range(99):
            yield self.default_transaction({'num' : i})
