from petal.pipeline.module_utils.OnlineLearner import OnlineLearner
from ..libraries.natural_language.topic_modeler import TopicModeler

import pickle

class TopicLDAModule(OnlineLearner):
    '''
    Model topics of articles in online batches
    '''
    def __init__(self, filename='data/models/topic_model.pkl'):
        OnlineLearner.__init__(self, in_label='Article', name='TopicLDAModule', filename=filename)

    def init_model(self):
        self.model = TopicModeler()
    
    def learn(self, model, batch):
        pass
