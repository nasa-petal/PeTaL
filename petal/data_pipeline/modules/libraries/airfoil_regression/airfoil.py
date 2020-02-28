from airfoil_dataset import AirfoilDataset
from airfoil_model   import AirfoilModel

from train import train

import os
import pickle
import torch
import torch.nn as nn

def run(model, testset):
    criterion = nn.MSELoss()
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)
    for inputs, labels in testloader:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(inputs)
            print(loss.item(), flush=True)

def load_dataset(picklefile='airfoil_dataset.pkl'):
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as infile:
            return pickle.load(infile)
    else:
        dataset = AirfoilDataset(filename='storage.h5')
        result = (*dataset.split(train=0.7), dataset.input_size(), dataset.output_size())
        with open(picklefile, 'wb') as outfile:
            pickle.dump(result, outfile)
        return result

def main():
    print('Loading dataset')
    trainset, testset, in_size, out_size = load_dataset()
    print('Finished loading dataset, length: ', len(trainset), ', ', len(testset))
    model         = AirfoilModel(in_size, out_size, hidden=2, width=150)
    model         = model.double()
    reverse_model = AirfoilModel(out_size, in_size, hidden=2, width=150)
    reverse_model = reverse_model.double()
    print('Loaded model, beginning training')
    do_training = True
    if do_training:
        train(model, reverse_model, trainset, n_epochs=1)
        torch.save(model.state_dict(), 'airfoil_model.pt')
        print('Finished Training', flush=True)
    else:
        model.load_state_dict(torch.load('airfoil_model.pt')) # Takes roughly .15s
        print('Loaded', flush=True)
    run(model, testset)


if __name__ == '__main__':
    main()
