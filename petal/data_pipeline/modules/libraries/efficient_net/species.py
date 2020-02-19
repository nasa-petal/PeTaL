from species_model import SpeciesModel

import json, os, os.path
from time import sleep, time

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from random import shuffle
from pprint import pprint

class Dataset(data.Dataset):
    def __init__(self, ids, labels):
        self.labels = labels
        self.list_IDs = ids

    def get_label(self, index):
        return self.list_IDs[index].split('_')[0].split('/')[-1]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        filename = ID
        tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
        print(filename)
        img = tfms(Image.open(filename))
        img = img.unsqueeze(0)
        x = img
        y = self.labels[ID.split('_')[0].split('/')[-1]]
        return x, y

def train(net, dataset, n_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # TODO: Change me later!
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    try:
        for epoch in range(n_epochs):
            print('Epoch: ', epoch, flush=True)
            running_loss = 0.0
            iterator = iter(trainloader)
            i = 0
            while True:
                try:
                    inputs, labels = next(iterator)
                    print('    datapoint: ', i, flush=True)
                    inputs = inputs.squeeze(dim=1)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    if i % 2000 == 1999:
                        print(' [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
                    i += 1
                except StopIteration:
                    break
                except RuntimeError as e:
                    print(e)
    except KeyboardInterrupt:
        pass
    return net

def run(model, image=None):
    model.eval()
    with torch.no_grad():
        outputs = model(image)

    print('-' * 80)
    print('')
    predictions = torch.topk(outputs, k=10).indices.squeeze(0).tolist()
    for idx in predictions:
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%'.format(label=idx, p=prob*100))
    print('')
    return predictions[0]

def build_dataset():
    image_dir = '../../../data/images/'
    counter = 0
    files   = []
    ids     = dict()
    for filename in os.listdir(image_dir):
        uuid     = filename.split('_')[0]
        filepath = image_dir + filename
        try:
            Image.open(filepath)
            if uuid not in ids:
                ids[uuid] = counter
                counter += 1
            files.append(filepath)
        except OSError as e:
            print(e)
            pass
    shuffle(files)
    cutoff = int(0.8 * len(files))
    print(len(files), cutoff)
    return Dataset(files[:cutoff], ids), Dataset(files[cutoff:], ids)

def main():
    trainset, testset = build_dataset()

    do_training = True
    # do_training = False
    PATH = 'species_net.pth'

    net = SpeciesModel()
    try:
        net.cuda()
    except AssertionError:
        print('*' * 80)
        print('Training on CPU!!!')
        print('*' * 80)
    if do_training:
        train(net, trainset, n_epochs=20)
        torch.save(net.state_dict(), PATH)
    else:
        start = time()
        net.load_state_dict(torch.load(PATH)) # Takes roughly .15s
        duration = time() - start
        print('Loading took: ', duration, 's')

    analysis = dict()
    for image, label in testset:
        index = run(net, image=image)
        predicted = testset.get_label(index)
        actual    = testset.get_label(label)
        if actual not in analysis:
            analysis[actual] = []
        print('Actual label:')
        print(label)
        # print('Predicted:')
        # print(index)
        analysis[actual].append(label == index)
    analysis = {k : round(sum(1 if t else 0 for t in v) / len(v), 2) for k, v in analysis.items()}
    pprint(analysis)

if __name__ == '__main__':
    main()
