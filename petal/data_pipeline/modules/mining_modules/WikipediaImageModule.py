from .ImageModule import ImageModule

class WikipediaImageModule(ImageModule):
    '''
    Download images from WikipediaArticle nodes in the neo4j database
    '''
    def __init__(self, in_label='WikipediaArticle', out_label='Image', connect_labels=('HAS_IMAGE', 'HAS_IMAGE'), name='WikipediaImages'):
        ImageModule.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, previous):
        # Lookup the species based on its name. Make sure that all Species objects have this attribute!!
        image_nodes = []
        
        title  = previous.data['title']
        images = previous.data['images']
        return ImageModule.process(self, images, uuid=previous.uuid, title=title)
