from ..libraries.encyclopedia_of_life.eol_image_scraper import get_images
from .ImageModule import ImageModule

class EOLImageModule(ImageModule):
    def __init__(self, in_label='Species', out_label='Image', connect_labels=('HAS_IMAGE', 'HAS_IMAGE'), name='EOL_Images'):
        ImageModule.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, node):
        name = node['name']
        pages = get_images(name)
        transactions = []
        for page in pages:
            for image_set in page:
                transactions.extend(ImageModule.process(self, image_set))
        return transactions
