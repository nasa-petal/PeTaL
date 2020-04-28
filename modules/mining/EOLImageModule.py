from .ImageModule import ImageModule
from ..libraries.encyclopedia_of_life.eol_image_scraper import get_images, get_media_page

class EOLImageModule(ImageModule):
    '''
    Download EOL images *WITHOUT* breaking any rules.
    '''
    def __init__(self, in_label='EOLPage', out_label='EOLImage:Image', connect_labels=('image', 'image'), name='EOLImages'):
        ImageModule.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, previous):
        '''
        Download images for a particular EOL page

        :param previous: neo4j transaction containing page_id property, donoting the page to scrape images from
        '''
        if 'page_id' in previous.data:
            page_id = previous.data['page_id']
            for gallery in get_media_page(page_id):
                for transaction in ImageModule.process(self, gallery, title=str(page_id), uuid=previous.uuid + '-image'):
                    transaction.from_uuid = previous.uuid
                    yield transaction
