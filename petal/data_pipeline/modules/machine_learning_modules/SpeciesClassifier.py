from ..utils.OnlineLearner import OnlineLearner
from ..libraries.efficient_net.species_model import SpeciesModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from torchvision import transforms
import os

class SpeciesClassifier(OnlineLearner):
    '''
    Classify species using a convolutional neural network
    '''
    def __init__(self, filename='data/models/species_classifier.nn'):
        OnlineLearner.__init__(self, in_label='Image', name='SpeciesClassifier', filename=filename)
        self.init_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9) # TODO: Change me later!

    def init_model(self):
        self.model = SpeciesModel()

    def save(self):
        torch.save(self.model.state_dict(), self.filename)

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.filename)) # Takes roughly .15s
        except RuntimeError:
            backup = self.filename + '.bak'
            if os.path.isfile(backup):
                os.remove(backup) # Removes old backup!
            os.rename(self.filename, backup)

    def learn(self, batch):
        for node in batch:
            print(node)
            # inputs, labels = data
            # inputs = inputs.squeeze(dim=1)

            # self.optimizer.zero_grad()
            # outputs = self.model(inputs)
            # loss = self.criterion(outputs, labels)
            # loss.backward()
            # self.optimizer.step()
