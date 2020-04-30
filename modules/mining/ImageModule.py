from petal.pipeline.utils.module import Module

import urllib.request
import urllib3
import requests

EXCLUDED_EXTENSIONS = {'svg', 'ogg'}

class ImageModule(Module):
    '''
    Base class for image handling within the PeTaL pipeline

    Creates an skeleton node with a filename reference to a *local* file.
    If you're distributing PeTaL across multiple servers, keep this in mind and create a dedicated image server, ideally close by (or even integrated with) the machine learning modules or other image-dependent apps.
    '''
    def __init__(self, in_label=None, out_label=None, connect_labels=None, name='AbstractImageModule'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, urls, uuid=None, title=None):
        '''
        Download a list of URLS and return skeleton neo4j nodes referencing the downloaded images.

        :param urls: Image urls to download (list)
        :param uuid: The uuid of what these images are of (i.e. "WikipediaArticle498_")
        :param title: The parent of these images, if a linkage is desired.
        '''
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
