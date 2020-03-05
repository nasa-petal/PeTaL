from ..utils.module import Module
from ..libraries.encyclopedia_of_life.eol_api import EOL_API

import requests

class EOLSpeciesModule(Module):
    def __init__(self, in_label=None, out_label='EOLPage', connect_labels=None, name='EOLSpecies', count=1900000):
        Module.__init__(self, in_label, out_label, connect_labels, name, count)

    def process(self):
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
                    if name is not None:
                        yield self.custom_transaction(data=properties, in_label='Taxon', out_label='EOLPage', uuid=name + '_eol_page', from_uuid=name, connect_labels=('eol_page', 'eol_page'))
                        # yield self.query_transaction('MATCH (t:Taxon {name:\'' + name + '\'}) SET t.eol_page = \'' + str(properties['page_id']) + '\' RETURN t')
                skip += page_size
            except KeyError as e:
                pass
            except requests.exceptions.SSLError as e:
                pass

