from .eol_api import EOL_API
from .module import Module

from pprint import pprint

class EOLModule(Module):
    def __init__(self, in_label='Species', out_label='EOLData', connect_labels=('MENTIONED_IN_DATA', 'MENTIONS_SPECIES')):
        Module.__init__(self, in_label, out_label, connect_labels)
        self.api = EOL_API()

    def process(self, node):
        print(node)
        name = node['name']
        pprint(self.api.search('MATCH (p:Page)-[:trait]->(t:Trait)-[:metadata]->(m) WHERE p.canonical=\'{name}\' RETURN m LIMIT 100'.format(name=name)))
        # pprint(self.api.search('MATCH (p:Page) WHERE p.canonical=\'{name}\' RETURN p.trait LIMIT 100'.format(name=name)))
        1/0
        return dict()
