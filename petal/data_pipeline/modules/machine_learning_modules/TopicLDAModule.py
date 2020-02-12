from ..module_utils.module import Module
from ..libraries.topic_modelling import TopicModeler

class TopicLDAModule(Module):
    '''
    Model topics of articles in online batches
    '''
    def __init__(self, in_label='Article', out_label=None, connect_labels=None, name='TopicLDAModule'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, node):
        print(node)
        1/0
