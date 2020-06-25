from .BatchLearner import BatchLearner

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

from time import sleep

class BatchTorchLearner(BatchLearner):
    '''
    Base class for pytorch machine learning modules
    For simple cases, transform() alone can be overloaded
    '''
    def __init__(self, criterion=None, optimizer=None, optimizer_kwargs=None, **kwargs):
        BatchLearner.__init__(self, **kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = criterion()
        if self.model is not None:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = None
        self.log.log('Calling base batch torch learner')


    def save(self):
        self.log.log('Saving model')
        torch.save(self.model.state_dict(), self.filename)

    def load(self):
        self.log.log('Loading model')
        try:
            self.model.load_state_dict(torch.load(self.filename)) # Takes roughly .15s
        except FileNotFoundError:
            self.log.log('Weight file {} not found, starting from scratch'.format(self.filename))

    def transform(self, node):
        '''
        Must yield a list of tuples of (inputs, labels) for training
        '''
        raise NotImplementedError('Must implement data transformation')

    def step(self, inputs, labels):
        raise RuntimeError('Batch learner called step()')

    def learn(self, batch):
        if self.optimizer is None:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self.log.log('Learning')
        input_list = []
        label_list = []
        for node in batch.items:
            for inputs, labels in self.transform(node):
                input_list.append(inputs)
                label_list.append(labels)
        if self.optimizer is None:
            self.optimizer = optimizer(self.model.parameters(), **self.optimizer_kwargs)
        self.optimizer.zero_grad()
        inputs = torch.cat(input_list)
        labels = torch.cat(label_list)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.log.log('{} loss: '.format(self.name), loss.item())
