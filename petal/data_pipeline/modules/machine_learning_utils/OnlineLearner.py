from ..module_utils.module import Module

import os, pickle

class OnlineLearner(Module):
    '''
    A dynamic machine learning module that supports online learning
    A subclass of this class needs to define a function learn(batch), save(model, filename) and load(filename), and init_model() 
    '''
    def __init__(self, in_label=None, name='OnlineLearner', filename=None):
        Module.__init__(self, in_label=in_label, out_label=None, connect_labels=None, name=name)
        self.filename = filename
        if (not self.filename is None) and os.path.isfile(self.filename):
            self.model = self.load(self.filename)
        else:
            self.init_model()

    def init_model(self):
        self.model = None
    
    def save(self, model, filename):
        print('Saving to pickle')
        with open(filename, 'wb') as outfile:
            pickle.dump(model, outfile)
    
    def load(self, filename):
        print('Loading from pickle')
        if os.path.isfile(filename):
            with open(filename, 'rb') as infile:
                return pickle.load(infile)
        else:
            print('Creating new ', self.name, ' model')
            self.init_model()
            return self.model

    def learn(self, model, batch):
        pass

    def process(self, node):
        print('Online learning')
        return
        self.model = self.load(self.filename)
        self.learn(model, node)
        self.save(model, filename)

