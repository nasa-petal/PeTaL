from ..libraries.encyclopedia_of_life.eol_image_api import get_images
from .ImageModule import ImageModule

class EOLModule(ImageModule):
    def __init__(self, in_label='Species', out_label='Image', connect_labels=('HAS_IMAGE', 'HAS_IMAGE'), name='EOL_Images'):
        ImageModule.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, node):
        name = node['name']
        images = get_images(name)
        return ImageModule.process(self, images)
