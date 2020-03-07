from .ImageModule import ImageModule
from ..libraries.encyclopedia_of_life.eol_image_scraper import get_images, get_media_page

class EOLImageModule(ImageModule):
    def __init__(self, in_label='Species', out_label='Image', connect_labels=('HAS_IMAGE', 'HAS_IMAGE'), name='EOL_Images'):
        ImageModule.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, previous):
        if 'page_id' in previous.data:
            page_id = previous.data['page_id']
            for gallery in get_media_page(page_id):
                for transaction in ImageModule.process(self, gallery, title=str(page_id), uuid=previous.uuid + '-image'):
                    print(page_id, flush=True)
                    transaction.from_uuid = previous.uuid
                    yield transaction
