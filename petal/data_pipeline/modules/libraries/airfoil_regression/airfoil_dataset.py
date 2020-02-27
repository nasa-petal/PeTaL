import torch.utils.data as data
import torch

import pandas as pd

class AirfoilDataset(data.Dataset):
    def __init__(self, filename='storage.h5'):
        self.outputs = []
        self.inputs = []
        for i in range(0,1020,20):
            self.outputs.append("y_ss_" + str(int(i)))
            self.outputs.append("y_ps_" + str(int(i)))
            self.inputs.append("cp_ss_" + str(int(i)))
            self.inputs.append("cp_ps_" + str(int(i)))
        self.inputs.extend(["Re", 'Ncrit', 'alpha'])
        self.outputs.extend(["Cl", 'Cd', 'Cdp', 'Cm']) # LABELS

        h5_file = 'storage.h5'
        df = pd.read_hdf(h5_file, 'data')
        self.df = df.drop(columns=['AirfoilName'])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        print(index)
        row = self.df.iloc[[index]]
        print(row)
        print(row.columns)
        return row[self.inputs].to_numpy(), row[self.outputs].to_numpy()

def split_dataset(dataset, train=0.7):
    l = len(dataset)
    train_size = int(l * train)
    test_size  = l - train_size
    return data.random_split(dataset, [train_size, test_size])

def main():
    dataset = AirfoilDataset()
    train, test = split_dataset(dataset, train=0.7)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    iterator = iter(trainloader)
    inputs, labels = next(iterator)
    print(inputs, labels)

if __name__ == '__main__':
    main()
