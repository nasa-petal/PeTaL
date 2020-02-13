from species_model import SpeciesModel

import json, os, os.path
from time import sleep

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms

class Dataset(data.Dataset):
    def __init__(self, ids, labels):
        self.labels = labels
        self.list_IDs = ids

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        filename = ID
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
        img = tfms(Image.open(filename))
        print(img.shape)
        img = img.unsqueeze(0)
        print(img.shape)
        x = img
        y = self.labels[ID.split('_')[0]]
        return x, y

def train(net, dataset, n_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # TODO: Change me later!
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    try:
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                print(i)
                inputs, labels = data
                inputs = inputs.squeeze(dim=1)
                # labels = labels.squeeze()
                print(inputs.shape)
                print(labels.shape)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 2000 == 1999:
                    print(' [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
    except KeyboardInterrupt:
        pass
    return net

# def run(model, image='img.jpg'):
#     tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
#     img = tfms(Image.open(image)).unsqueeze(0)
#     print(img.shape)
# 
#     model.eval()
#     with torch.no_grad():
#         outputs = model(img)
# 
#     for idx in torch.topk(outputs, k=10).indices.squeeze(0).tolist():
#         prob = torch.softmax(outputs, dim=1)[0, idx].item()
#         print('{label:<75} ({p:.2f}%'.format(label=idx, p=prob*100))

def main():
    dataset = Dataset(['../../../data/images/35bfa85e-16d3-4942-b57b-da48107e69ba_8.jpg'] * 2, {'../../../data/images/35bfa85e-16d3-4942-b57b-da48107e69ba' : 0})#torch.zeros(68)})

    do_training = True
    PATH = 'species_net.pth'

    net = SpeciesModel()
    if do_training:
        train(net, dataset)
        torch.save(net.state_dict(), PATH)
    else:
        net.load_state_dict(torch.load(PATH))
    run(net, image='../../../data/images/35bfa85e-16d3-4942-b57b-da48107e69ba_8.jpg')

if __name__ == '__main__':
    main()
