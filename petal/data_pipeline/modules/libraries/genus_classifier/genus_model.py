# Edited by Lucas Saldyt, adapted from:
# https://github.com/lukemelas/EfficientNet-PyTorch
# Pytorch model was chosen for ease of use and to extend the capabilities of PeTaL to support both libraries

from efficientnet_pytorch import EfficientNet as EfficientNetBase

import json, os, os.path

from time import sleep
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

class GenusModel(nn.Module):
    def __init__(self, i=0, classes=2):
        nn.Module.__init__(self)
        if i < 0 or i > 7:
            raise ValueError('Parameter i to Efficient Net Model must be between 0 and 7 inclusive, but was: {}'.format(i))
        # Top-1 Accuracy ranges from 76.3% to 84.4%, in intervals of roughly 1-2% between indexes
        self.feature_extractor = EfficientNetBase.from_pretrained('efficientnet-b{}'.format(i)) # Can go up to b7, with b0 having the least parameters, and b7 having the most (but more accuracy)
        self.fc1 = nn.Linear(1280 * 7 * 7, classes * 10)
        self.fc2 = nn.Linear(classes * 10, classes)

    def forward(self, x):
        x = self.feature_extractor.extract_features(x)
        x = x.view(1280 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.unsqueeze(dim=0)
        return x
