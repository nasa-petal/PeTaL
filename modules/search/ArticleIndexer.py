from bitflow.utils.module import Module
from ..libraries.natural_language.cleaner import Cleaner
from ..libraries.natural_language.hitlist import HitList

class ArticleIndexer(Module):
    '''
    This module is intended to index articles within PeTaL.

    The bulk of this is handled by the HitList class.

    Can parse any article with "title", "abstract" and "content" properties
    '''
    def __init__(self, in_label='Article', out_label='HitList', connect_labels=('hitlist', 'hitlist'), name='ArticleIndexer'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=True)
        self.SECTIONS = {'title', 'abstract', 'content'}
        self.cleaner = Cleaner()

    def process(self, previous):
        '''
        Generate a hitlist for a particular article.

        :param previous: Transaction object of a particular article
        '''
        self.log.log('Running Indexer')

        hitlist = HitList(previous.uuid)

        data = previous.data
        for section in self.SECTIONS:
            try:
                text = data[section]
                for word in self.cleaner.clean(text):
                    hitlist.add(section, word)
            except KeyError:
                print('Warning: article does not contain parsed content: "{}", cannot be indexed'.format(section))

        try:
            hitlist.save()
            yield self.default_transaction(data=dict(filename=hitlist.filename, source_uuid=str(previous.uuid)), uuid=str(previous.uuid) + '_hitlist')
        except OSError as e:
            print(e)
