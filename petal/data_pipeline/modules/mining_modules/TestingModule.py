from ..module_utils.module import Module

class TestingModule(Module):
    '''
    This module populates neo4j with Species nodes, allowing WikipediaModule and others to process them.
    Notice how BackboneModule's in_label is None, which specifies that it is independent of other neo4j nodes
    '''
    def __init__(self, in_label=None, out_label='Species', connect_label=None, name='Testing', count=1):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self):
        # Load a SINGLE species into the database, and process it.
        yield self.default_transaction({'name' : 'Encephalartos horridus'})
