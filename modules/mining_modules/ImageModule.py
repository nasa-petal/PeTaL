from petal.pipeline.module_utils.module import Module

import urllib.request
import urllib3
import requests

EXCLUDED_EXTENSIONS = {'svg', 'ogg'}

class ImageModule(Module):
    def __init__(self, in_label=None, out_label=None, connect_labels=None, name='AbstractImageModule'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, urls, uuid=None, title=None):
        if title is None:
            title = 'independent'
        for i, image in enumerate(urls):
            ext = image.split('.')[-1]
            if ext not in EXCLUDED_EXTENSIONS:
                filename = 'data/images/{uuid}_{i}.'.format(uuid=str(uuid), i=str(i)) + ext
                try:
                    yield self.default_transaction(data=dict(filename=filename, url=image, parent=title), uuid=uuid + '-' + str(i))
                    urllib.request.urlretrieve(image, filename)
                except urllib3.exceptions.NewConnectionError:
                    pass
                except TimeoutError:
                    pass
                except urllib.error.HTTPError:
                    pass
                except requests.exceptions.ConnectionError:
                    pass
