from petal.pipeline.utils.BatchTorchLearner import BatchTorchLearner

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms

import json, os, os.path, pickle

from time import sleep
from PIL import Image
import pickle
import plotly.graph_objects as go

'''
A simple neural net for edge detection
'''

DPI = 400

def smooth(array, amount):
    '''
    Sample an array by averaging every N elements
    Useful in creating smaller vector inputs and outputs for faster machine learning iteration

    :param array: Array to sample from (python list or numpy array)
    :param amount: N, i.e. 5 for an array of length 100 to create 20 samples.
    '''
    new = []
    running = 0
    for i, x in enumerate(array):
        running += x
        if i % amount == 0:
            new.append(running/amount)
            running = 0
    return new

class EdgeRegressorModel(nn.Module):
    '''
    Regress an airfoils shape from an image of the airfoil

    (Pytorch module)
    '''
    def __init__(self, depth=1, activation=nn.ReLU, out_size=120, mid_channels=8):
        '''
        :param depth: Number of convolution layers to use
        :param activation: Pytorch activation to use, ReLU by default
        :param out_size: Expected size of output vector, encoding pressure and suction sides of an airfoil
        :param mid_channels: Channels used by inner conv layers
        '''
        nn.Module.__init__(self)
        self.conv_layers = []
        for i in range(depth):
            if i == 0:
                channels = 4
            else:
                channels = 8
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(channels, mid_channels, 3, padding=1),
                activation()
                ))
            if i < 2:
                self.conv_layers.append(nn.MaxPool2d(2))
        self.prefinal = nn.Sequential(
            nn.Linear(56 * 56 * mid_channels, 1000),
            )
        self.final = nn.Sequential(nn.Linear(1000, out_size))

    def forward(self, x):
        '''
        :param x: Input vector representing an encoded image
        '''
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, 56 * 56 * 8)
        x = self.prefinal(x)
        x = self.final(x)
        x = x.double()
        return x

class AirfoilEdgeRegressor(BatchTorchLearner):
    '''
    Regress an airfoils shape from an image of the airfoil

    (Pipeline module)
    '''
    def __init__(self, filename='data/models/airfoil_edge_regressor.nn', name='AirfoilEdgeRegressor'):
        '''
        :param filename: The location to save the model parameters to
        :param name: What to call this module..
        '''
        BatchTorchLearner.__init__(self, filename=filename, epochs=2, train_fraction=0.8, test_fraction=0.2, validate_fraction=0.00, criterion=nn.MSELoss, optimizer=optim.Adadelta, optimizer_kwargs=dict(lr=1.0, rho=0.9, eps=1e-06, weight_decay=0), in_label='AugmentedAirfoilPlot', name=name)
        self.log.log('Created AirfoilEdgeRegressor')

    def load_image(self, filename):
        '''
        :param filename: (str)
        '''
        self.log.log('Loading Image ', filename)
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        image = Image.open(filename)
        image.putalpha(255)
        img = tfms(image)
        img = img.unsqueeze(0)
        return img

    def load_labels(self, parent):
        '''
        Load coordinate labels given the UUID of a neo4j node in the database

        :param parent: UUID of existing neo4j node with 'coord_file' property
        '''
        self.log.log('Loading Labels')
        parent = self.driver.get(parent)
        with open(parent['coord_file'], 'rb') as infile:
            coordinates = pickle.load(infile)
        fx, fy, sx, sy, camber = coordinates
        fx = [x for i, x in enumerate(fx) if i % 5 == 0]
        fy = [y for i, y in enumerate(fy) if i % 5 == 0]
        sy = [y for i, y in enumerate(sy) if i % 5 == 0]
        coordinates = sum(map(list, [fx, fy, sy]), [])
        labels = torch.tensor(coordinates, dtype=torch.double)
        return labels.unsqueeze(0)

    def init_model(self):
        self.log.log('Initializing Module')
        self.model = EdgeRegressorModel(depth=3)

    # def learn() inherited, uses transform()
    def transform(self, node):
        '''
        Provide data to the ML model to be trained.

        :param node: 
        '''
        try:
            self.log.log('Transforming')
            labels = self.load_labels(node.data['parent'])
            image  = self.load_image(filename = node.data['filename'])
            self.log.log('Yielding data')
            yield image, labels
        except OSError as e:
            self.log.log(e)

    def test(self, batch):
        '''
        Test procedure on a batch of input images

        :param batch: Batch object holding neo4j Image() nodes
        '''
        self.log.log('Testing on ', batch.uuid)
        for item in batch.items:
            data        = item.data
            filename    = data['filename']
            parent      = self.driver.get(data['parent'])
            with open(parent['coord_file'], 'rb') as infile:
                coordinates = pickle.load(infile)
            image       = self.load_image(filename)
            predicted = self.model(image).detach().numpy()[0]
            self.plot(coordinates, predicted, filename=filename.replace('.png', '_regression.html'))

    def val(self, batch):
        '''
        Validation procedure on a batch of input images

        :param batch: Batch object holding neo4j Image() nodes
        '''
        raise NotImplementedError('AirfoilEdgeRegressor does not use validation yet')
        self.log.log('Validating on ', batch.uuid)

    def plot(self, coordinates, predicted, filename=None):
        '''
        Plot airfoil coordinates with plotly

        :param batch: Batch object holding neo4j Image() nodes
        '''
        fig = go.Figure()
        fx, fy, sx, sy, camber = coordinates
        fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='top'))
        fig.add_trace(go.Scatter(x=fx, y=sy, mode='lines', name='bottom'))

        fx = predicted[:40]
        fy = predicted[40:80]
        sy = predicted[80:120]
        fx = smooth(fx, 2)
        fy = smooth(fy, 2)
        sy = smooth(sy, 2)

        fig.add_trace(go.Scatter(x=fx, y=fy, mode='lines', name='top'))
        fig.add_trace(go.Scatter(x=fx, y=sy, mode='lines', name='bottom'))
        fig.update_layout(title='Edge Regression Test')
        fig.write_html(filename, auto_open=False)
        self.log.log('Wrote Airfoil plot to file ', filename)
