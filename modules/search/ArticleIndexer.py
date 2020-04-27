from petal.pipeline.module_utils.module import Module
from ..libraries.natural_language.cleaner import Cleaner
from ..libraries.natural_language.hitlist import HitList

class ArticleIndexer(Module):
    '''
    This module is intended to index articles within PeTaL
    '''
    def __init__(self, in_label='Article', out_label='HitList', connect_labels=('hitlist', 'hitlist'), name='ArticleIndexer'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=True)
        self.SECTIONS = {'title', 'abstract', 'content'}
        self.cleaner = Cleaner()

    def process(self, previous):
        self.log.log('Running Indexer')

        hitlist = HitList(previous.uuid)

        data = previous.data
        for section in self.SECTIONS:
            text = data[section]
            for word in self.cleaner.clean(text):
                hitlist.add(section, word)

        try:
            hitlist.save()
            yield self.default_transaction(data=dict(filename=hitlist.filename, source_uuid=str(previous.uuid)), uuid=str(previous.uuid) + '_hitlist')
        except OSError as e:
            print(e)
