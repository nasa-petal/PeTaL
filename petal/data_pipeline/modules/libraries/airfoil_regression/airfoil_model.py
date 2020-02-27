import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F

class AirfoilModel(nn.Module):
    def __init__(self, inputs, outputs, hidden=1, width=100):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.fc1 = nn.Linear(inputs, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, outputs)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        for i in range(self.hidden):
            x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x.unsqueeze(dim=0)
