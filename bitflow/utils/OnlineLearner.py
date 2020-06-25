from .module import Module

import os, pickle

class OnlineLearner(Module):
    '''
    A dynamic machine learning module that supports online learning
    A subclass of this class needs to define a function learn(batch), save(model, filename) and load(filename), and init_model() 
    '''
    def __init__(self, filename=None, **kwargs):
        Module.__init__(self, **kwargs)
        self.filename = filename
        self.model = None
        if os.path.isfile(self.filename):
            self.load()

    def init_model(self):
        self.model = None
    
    def save(self):
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.model, outfile)
    
    def load(self):
        with open(self.filename, 'rb') as infile:
            self.model = pickle.load(infile)

    def learn(self, batch):
        pass

    def process(self, node):
        if self.model is None:
            self.init_model()
        if os.path.isfile(self.filename):
            self.load()
        self.learn(node)
        self.save()
        return []

