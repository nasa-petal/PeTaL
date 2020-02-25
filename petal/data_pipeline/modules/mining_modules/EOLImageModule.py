from .ImageModule import ImageModule
from ..libraries.encyclopedia_of_life.eol_image_scraper import get_images

class EOLImageModule(ImageModule):
    def __init__(self, in_label='Species', out_label='Image', connect_labels=('HAS_IMAGE', 'HAS_IMAGE'), name='EOL_Images'):
        ImageModule.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, previous):
        name = previous.data['name']
        pages = get_images(name)
        for page in pages:
            for image_set in page:
                for transaction in ImageModule.process(self, image_set, title=name, uuid=previous.uuid + '-image'):
                    transaction.from_uuid = previous.uuid
                    yield transaction
