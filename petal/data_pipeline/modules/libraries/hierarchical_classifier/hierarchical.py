from hierarchical_model import HierarchicalModel
from neo4j import GraphDatabase, basic_auth

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

    # def get_label(self, index):
    # return self.list_IDs[index].split('_')[0].split('/')[-1]

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

                    print(labels)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    print(outputs)
                    loss = sum(criterion(suboutput, sublabel) for suboutput, sublabel in zip(outputs, labels))
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
                except OSError as e:
                    print(e)
    except KeyboardInterrupt:
        pass
    return net

def run(model, image=None):
    model.eval()
    with torch.no_grad():
        outputs = model(image)

    # print('-' * 80)
    # print('')
    all_preds = []
    for suboutput in outputs:
        predictions = torch.topk(suboutput, k=2).indices.squeeze(0).tolist()
        for idx in predictions:
            prob = torch.softmax(suboutput, dim=1)[0, idx].item()
        all_preds.append(predictions[0])
        # print('{label:<75} ({p:.2f}%'.format(label=idx, p=prob*100))
    # print('')
    print('.', end='')
    return all_preds

def build_dataset():
    neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"), encrypted=False)
    image_dir = '../../../data/images/'
    files   = []
    # Yes this code is messy please clean it soon thx
    enc_id = 0
    alo_id = 0
    enc_ids = dict()
    alo_ids = dict()
    ids     = dict()
    index = 0
    for filename in os.listdir(image_dir):
        uuid     = filename.split('_')[0]
        with neo_client.session() as session:
            records = list(session.run('match (s) where s.uuid = \'{uuid}\' return s.parent'.format(uuid=uuid)).records())
            if len(records) == 0:
                # print(uuid)
                continue
            genus, sub = records[0]['s.parent'].strip().split(' ')
            if genus == 'Encephalartos':
                if sub in enc_ids:
                    y = enc_ids[sub]
                else:
                    y = enc_id
                    enc_ids[sub] = y
                    enc_id += 1
                ids[uuid] = (0, y)
            else:
                if sub in alo_ids:
                    y = alo_ids[sub]
                else:
                    y = alo_id
                    alo_ids[sub] = y
                    alo_id += 1
                ids[uuid] = (1, y)
        filepath = image_dir + filename
        files.append(filepath)
        index += 1
    shuffle(files)
    cutoff = int(0.8 * len(files))
    print(len(files), cutoff)
    return Dataset(files[:cutoff], ids), Dataset(files[cutoff:], ids)

def main():
    trainset, testset = build_dataset()

    do_training = True
    do_training = False
    PATH = 'species_net.pth'

    net = HierarchicalModel(i=0)
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

    total   = len(testset)
    correct = 0
    genus = 0
    species = 0
    for image, label in testset:
        index = run(net, image=image)
        print(index, label)
        # predicted = testset.get_label(tuple(index))
        # actual    = testset.get_label(tuple(label))
        if tuple(index) == tuple(label):
            correct += 1
        if index[0] == label[0]:
            genus += 1
        if index[1] == label[1]:
            species += 1
    print('OVERALL ACCURACY')
    print(correct / total)
    print('Genus level:')
    print(genus / total)
    print('Species level:')
    print(species / total)

if __name__ == '__main__':
    main()
