from airfoil_dataset import AirfoilDataset
from airfoil_model   import AirfoilModel

from train import train

def main():
    print('Loading dataset')
    dataset = AirfoilDataset(filename='storage.h5')
    print('Finished loading dataset, length:', len(dataset))
    model   = AirfoilModel(dataset.input_size(), dataset.output_size(), hidden=1, width=100)
    model   = model.double()
    trainset, testset = dataset.split(train=0.7)
    print('Loaded model, beginning training')
    train(model, trainset, n_epochs=2)

if __name__ == '__main__':
    main()
