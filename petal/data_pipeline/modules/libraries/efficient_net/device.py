import torch
from species_model import SpeciesModel

device = torch.device('cuda:0')
print(device)

model = SpeciesModel()
print(model.to(device))
