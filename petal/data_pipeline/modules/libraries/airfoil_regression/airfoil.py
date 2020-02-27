from airfoil_dataset import AirfoilDataset
from airfoil_model   import AirfoilModel

from train import train

def main():
    dataset = AirfoilDataset(filename='storage.h5')
    model   = AirfoilModel(dataset.input_size(), dataset.output_size(), hidden=1, width=100)
    model   = model.double()
    trainset, testset = dataset.split(train=0.7)
    train(model, trainset, n_epochs=2)

if __name__ == '__main__':
    main()
