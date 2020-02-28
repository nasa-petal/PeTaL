from ..utils.module import Module
from ..libraries.encyclopedia_of_life.eol_api import EOL_API
from ..libraries.catalog import build_catalog

import requests

class EOLSpeciesModule(Module):
    '''
    '''
    def __init__(self, in_label=None, out_label='Species:Taxon', connect_label=None, name='EOLSpecies', count=1900000):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self):
        # catalog = build_catalog()
        api = EOL_API()
        page_size = 1000
        skip  = 0
        while True:
            try:
                results = api.search('MATCH (x:Page) RETURN x.canonical, x.page_id, x.rank skip {skip} limit {limit}'.format(skip=skip, limit=page_size))
                keys = results['columns']
                page = results['data']
                for values in page:
                    properties = dict()
                    for key, value in zip(keys, values):
                        key = key.replace('x.', '')
                        properties[key] = value
                    name = properties['canonical']
                    properties = {k : v if v is not None else 'None' for k, v in properties.items()}
                    # properties.update(catalog[name])
                    # print(name, flush=True)
                    yield self.default_transaction(properties, uuid=name)
                skip += page_size
            except KeyError as e:
                # print(e, flush=True)
                pass
            except requests.exceptions.SSLError as e:
                # print('SSL Error: ', e, flush=True)
                pass
