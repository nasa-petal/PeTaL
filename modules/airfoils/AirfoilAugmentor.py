import numpy as np
from time import sleep
from PIL import Image, ImageDraw, ImageOps

from random import randint, random

from petal.pipeline.module_utils.module import Module

class AirfoilAugmentor(Module):
    def __init__(self, count=100):
        Module.__init__(self, in_label='CleanAirfoilPlot', out_label='AugmentedAirfoilPlot', connect_labels=('augmented_image', 'augmented_image'))
        self.count = count

    def random_color(self):
        return tuple(randint(0, 255) for _ in range(4))

    def noise(self, image, p=1.0):
        width, height = image.size
        noise_map = np.random.randint(int(p*255), size=(height, width, 4,), dtype='uint8')
        image += noise_map
        return Image.fromarray(image)

    def rand_fill(self, image):
        width, height = image.size
        center = int(0.5 * width), int(0.5 * height)
        origin = 0, 0

        ImageDraw.floodfill(image, xy=center, value=self.random_color())
        ImageDraw.floodfill(image, xy=origin, value=self.random_color())
        return image

    def white_edge_fill(self, image):
        white = (255, 255, 255, 255)
        width, height = image.size
        for xy in [(1, 1), (1, height - 1), (width - 1, 1), (width - 1, height - 1)]:
            ImageDraw.floodfill(image, xy=xy, value=white)
        return image

    def rand_affine(self, image):
        horizontal = randint(-15, 15)
        vertical   = randint(-5, 5)
        stretch_x  = 0.9 + 0.2 * random() # 0.9 to 1.1
        stretch_y  = 0.9 + 0.2 * random()

        # x, y ->
        # a x + by + c, d x + e y + f
        return image.transform(image.size, Image.AFFINE, (stretch_x, 0, horizontal, 0, stretch_y, vertical))

    def flips(self, image):
        return [image] # + list(map(Image.fromarray, [np.fliplr(image), np.flipud(image), np.fliplr(np.flipud(image))]))

    def augment(self, filename):
        image = Image.open(filename)
        for j in range(self.count):
            for i, flipped in enumerate(self.flips(image)):
                aug_image = ImageOps.expand(flipped, (60,) * 4)
                aug_image = aug_image.rotate(randint(-5, 5))
                aug_image = self.rand_affine(aug_image)
                aug_image = self.white_edge_fill(aug_image)
                # aug_image = self.rand_fill(aug_image)
                # aug_image = self.noise(aug_image, p=0.01)

                aug_file = filename.replace('.png', '_augmented_{}_{}.png'.format(i, j))
                aug_image.save(aug_file)
                yield aug_file

    def process(self, node):
        for filename in self.augment(node.data['filename']):
            yield self.default_transaction(data=dict(filename=filename, parent=node.data['parent']))
