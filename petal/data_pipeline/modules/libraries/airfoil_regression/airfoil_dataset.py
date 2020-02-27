import torch.utils.data as data
import torch

from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

def change_type(o):
    if isinstance(o, np.int64): 
        return int(o)  
    return o

import numpy as np
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

        df = pd.read_hdf(filename, 'data')
        self.df = df.drop(columns=['AirfoilName'])
        self.df = self.normalize(df, self.inputs + self.outputs)

    def normalize(self, df, columns):
        min_max_scaler = MinMaxScaler()
        min_max_feature = {}
        for col in columns:
            min_F = change_type(df[[col]].min()[0])
            max_F = change_type(df[[col]].max()[0])
            
            min_max_feature[col] = [min_F,max_F]
            feat_scaled = min_max_scaler.fit_transform(df[[col]].values.astype(float))
            df[[col]] = feat_scaled
        return df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        row = self.df.iloc[[index]]
        inputs  = row[self.inputs].to_numpy(dtype=np.double)
        outputs = row[self.outputs].to_numpy(dtype=np.double)
        return inputs, outputs

    def input_size(self):
        return len(self.inputs)

    def output_size(self):
        return len(self.outputs)

    def split(self, train=0.7):
        l = len(self)
        train_size = int(l * train)
        test_size  = l - train_size
        return data.random_split(self, [train_size, test_size])

def main():
    dataset = AirfoilDataset()
    train, test = dataset.split(train=0.7)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    iterator = iter(trainloader)
    inputs, labels = next(iterator)
    print(inputs, labels)

if __name__ == '__main__':
    main()
