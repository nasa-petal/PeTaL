from ..utils.OnlineLearner import OnlineLearner
from ..libraries.efficient_net.species_model import SpeciesModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from PIL import Image

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
        self.labels = dict()
        self.index  = 0

    def load_image(self, filename):
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
        img = tfms(Image.open(filename))
        img = img.unsqueeze(0)
        return img

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

    def learn(self, node):
        try:
            species  = node['parent']
            filename = node['filename']
            image    = self.load_image(filename)
            
            if species in self.labels:
                labels = self.labels[species]
            else:
                labels = self.index
                self.labels[species] = self.index
                self.index += 1

            labels = torch.tensor([labels], dtype=torch.long)
            inputs = image.squeeze(dim=1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        except RuntimeError:
            pass
        except OSError:
            pass
