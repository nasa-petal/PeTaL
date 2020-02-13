from ..module_utils.module import Module

from uuid import uuid4

import urllib.request
import urllib3
import requests

EXCLUDED_EXTENSIONS = {'svg'}

class WikipediaImageModule(Module):
    '''
    Download images from WikipediaArticle nodes in the neo4j database
    '''
    def __init__(self, in_label='WikipediaArticle', out_label='Image', connect_labels=('HAS_IMAGE', 'HAS_IMAGE'), name='WikipediaImages'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, node):
        # Lookup the species based on its name. Make sure that all Species objects have this attribute!!
        image_nodes = []
        
        title  = node['title']
        images = node['images']
        for i, image in enumerate(images):
            ext = image.split('.')[-1]
            if ext not in EXCLUDED_EXTENSIONS:
                filename = 'data/images/{uuid}_{i}.'.format(uuid=str(node['uuid']), i=str(i)) + ext
                try:
                    urllib.request.urlretrieve(image, filename)
                except urllib3.exceptions.NewConnectionError:
                    pass
                except TimeoutError:
                    pass
                except requests.exceptions.ConnectionError:
                    pass
                image_nodes.append(self.default_transaction(data=dict(filename=filename, url=image, parent=title)))
        return image_nodes
