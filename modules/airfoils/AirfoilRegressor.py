from petal.pipeline.module_utils.BatchTorchLearner import BatchTorchLearner

import pickle
import os
import math

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

class AirfoilModel(nn.Module):
    def __init__(self, inputs, outputs, hidden=1, width=1000):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.fc1 = nn.Linear(inputs, width)
        self.inner = [nn.Linear(width, width) for i in range(hidden)]
        self.fc3 = nn.Linear(width, outputs)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        for inner in self.inner:
            x = F.selu(inner(x))
        x = self.fc3(x)
        return x

class AirfoilRegressor(BatchTorchLearner):
    '''
    Regress the performance of airfoil geometries
    Implements the `Module` interface, which requires a type signature and process() function.
    The `OnlineTorchLearner` class is defined in `utils`, and specifies common operations of online machine learning models
    To inherit from this class, AirfoilRegressor must specify `init_model`, and `transform`.
    Also, a filename can be specified to the parent constructor to specify where the model is saved.
    '''
    def __init__(self, name='AirfoilRegressor', filename='data/models/airfoil_regressor.nn'):
        # Take Airfoils as input, and produce no outputs.
        optimizer_kwargs = dict(lr=0.0001, momentum=0.9)
        BatchTorchLearner.__init__(self, nn.MSELoss, optim.SGD, optimizer_kwargs, in_label='Airfoil', out_label=None, name=name, filename=filename, train_fraction=0.9, validate_fraction=0.05, test_fraction=0.05)

    def init_model(self):
        self.model = AirfoilModel(1000 + 3 + 3, 4, hidden=5)

    def read_node(self, node):
        coord_file  = node.data['coord_file']
        detail_files = node.data['detail_files']

        for detail_file in detail_files:
            with open(coord_file, 'rb') as infile:
                coordinates = pickle.load(infile)
            with open(detail_file, 'rb') as infile:
                details = pickle.load(infile)

            signed_log = lambda x : 0 if x == 0 else math.copysign(math.log(abs(x)), x)

            mach  = signed_log(node.data['mach'])
            Re    = signed_log(node.data['Re'])
            Ncrit = signed_log(node.data['Ncrit'])
            regime_vec = [mach, Re, Ncrit]

            coefficient_tuples = list(zip(*(details[k] for k in sorted(details.keys()) if k.startswith('C'))))
            coefficient_keys   = [k for k in sorted(details.keys()) if k.startswith('C')]
            alphas = details['alpha']
            limits = list(zip(details['Top_Xtr'], details['Bot_Xtr']))
            yield coordinates, coefficient_tuples, coefficient_keys, alphas, limits, regime_vec

    def transform(self, node):
        for coordinates, coefficient_tuples, coefficient_keys, alphas, limits, regime_vec in self.read_node(node):
            coordinates = sum(map(list, coordinates), [])
            for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
                coefficients = torch.Tensor(coefficients)
                inputs       = torch.Tensor(coordinates + regime_vec + [top, bot, alpha])
                yield inputs.unsqueeze(0), coefficients.unsqueeze(0)

    def test(self, batch):
        self.log.log('Testing AirfoilRegressor')
        # for node in batch.items:
        #     for coordinates, coefficient_tuples, coefficient_keys, alphas, limits, regime_vec in self.read_node(node):
        #         coordinates = sum(map(list, coordinates), [])
        #         for alpha, coefficients, (top, bot) in zip(alphas, coefficient_tuples, limits):
        #             coefficients = torch.Tensor(coefficients)
        #             inputs       = torch.Tensor(coordinates + regime_vec + [top, bot, alpha])
        #             with torch.no_grad():
        #                 metrics = self.model(inputs)
        #             data = dict()
        #             for i, key in enumerate(coefficient_keys):
        #                 data[key] = metrics[i].item()
        #             # yield self.default_transaction(data=data)
