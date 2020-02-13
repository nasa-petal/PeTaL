# Edited by Lucas Saldyt, adapted from:
# https://github.com/lukemelas/EfficientNet-PyTorch
# Pytorch model was chosen for ease of use and to extend the capabilities of PeTaL to support both libraries

from efficientnet_pytorch import EfficientNet as EfficientNetBase

import json, os, os.path
from time import sleep

from PIL import Image
import torch
from torchvision import transforms

class EfficientNetModel:
    def __init__(self, i=0):
        if i < 0 or i > 7:
            raise ValueError('Parameter i to Efficient Net Model must be between 0 and 7 inclusive, but was: {}'.format(i))
        # Top-1 Accuracy ranges from 76.3% to 84.4%, in intervals of roughly 1-2% between indexes
        self.model = EfficientNetBase.from_pretrained('efficientnet-b{}'.format(i)) # Can go up to b7, with b0 having the least parameters, and b7 having the most (but more accuracy)

    def run(self, image='img.jpg'):
        tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]) # Explanation of these magic numbers??
        img = tfms(Image.open(image)).unsqueeze(0)
        print(img.shape)

        labels_map = json.load(open('labels_map.txt'))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img)

        for idx in torch.topk(outputs, k=10).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%'.format(label=labels_map[idx], p=prob*100))
    

if __name__ == '__main__':
    model = EfficientNetModel(i=0)
    for image in os.listdir('../../../data/images/'):
        try:
            print(image, flush=True)
            model.run(image='../../../data/images/' + image)
            sleep(1)
        except RuntimeError:
            pass
    # model.run(image='test2.jpg')
    # model.run(image='birds.jpg')
    # model.run(image='birdie.jpg')
