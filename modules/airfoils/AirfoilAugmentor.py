import numpy as np
from time import sleep
from PIL import Image, ImageDraw, ImageOps

from random import randint, random

from bitflow.utils.module import Module

'''
This module augments airfoils and plots the augmented images, for input into further machine learning modules

Augmentation includes translation, rotation, random noise, mirroring, and random color fills inside of and outside of shapes.
'''

class AirfoilAugmentor(Module):
    '''
    A module that takes airfoil images and augments them. 
    Can be easily adapted for any kind of image.
    '''
    def __init__(self, count=100):
        '''
        :param count: Number of augmentations to do
        '''
        Module.__init__(self, in_label='CleanAirfoilPlot', out_label='AugmentedAirfoilPlot', connect_labels=('augmented_image', 'augmented_image'))
        self.count = count

    def random_color(self):
        return tuple(randint(0, 255) for _ in range(4))

    def noise(self, image, p=1.0):
        '''
        Add random noise to the pixels in an image 

        :param image: PIL image to augment
        :param p: probability of noise, 0.0 to 1.0
        '''
        width, height = image.size
        noise_map = np.random.randint(int(p*255), size=(height, width, 4,), dtype='uint8')
        image += noise_map
        return Image.fromarray(image)

    def rand_fill(self, image):
        '''
        Fill the center and edge of an image with two separate randomly picked colors.
        For airfoils, since the airfoil is an enclosed shape, this makes the background and cross section colors irrelevant to the regressor as signals

        :param image: PIL image to augment
        '''
        width, height = image.size
        center = int(0.5 * width), int(0.5 * height)
        origin = 0, 0

        ImageDraw.floodfill(image, xy=center, value=self.random_color())
        ImageDraw.floodfill(image, xy=origin, value=self.random_color())
        return image

    def white_edge_fill(self, image):
        '''
        Fill the edges of an image with white, useful after rotating or other operations

        :param image: PIL image
        '''
        white = (255, 255, 255, 255)
        width, height = image.size
        for xy in [(1, 1), (1, height - 1), (width - 1, 1), (width - 1, height - 1)]:
            ImageDraw.floodfill(image, xy=xy, value=white)
        return image

    def rand_affine(self, image, h=15, v=5, stretch=0.9):
        '''
        Perform an affine transform

        :param h: Pixels to shift the image horizontally by, randomly chosen from (-h, h)
        :param v: Pixels to shift the image vertically by, randomly chosen from (-v, v)
        :param stretch: Stretching factor, both vertical and horizontal (shared)
        '''
        horizontal = randint(-h, h)
        vertical   = randint(-v, v)
        stretch_x  = stretch + 2.0 * (1.0 - stretch) * random()
        stretch_y  = stretch + 2.0 * (1.0 - stretch) * random()

        # x, y ->
        # a x + by + c, d x + e y + f
        return image.transform(image.size, Image.AFFINE, (stretch_x, 0, horizontal, 0, stretch_y, vertical))

    def flips(self, image):
        '''

        Return all four flipped variants of an image: Original, mirrored left-right, mirrored left-right and up-down, and mirrored up-down

        :param image: PIL image object
        '''
        return [image] + list(map(Image.fromarray, [np.fliplr(image), np.flipud(image), np.fliplr(np.flipud(image))]))

    def augment(self, filename):
        '''
        Perform augmentations on a source file

        :param filename: string pointing to an image openable by PIL
        '''
        image = Image.open(filename)
        for j in range(self.count):
            for i, flipped in enumerate(self.flips(image)):

                # The order here *does* matter. 
                # White edge fill needs to follow rand_affine
                # TODO move these parameters to the class initialization?
                aug_image = ImageOps.expand(flipped, (60,) * 4)
                aug_image = aug_image.rotate(randint(-5, 5))
                aug_image = self.rand_affine(aug_image)
                aug_image = self.white_edge_fill(aug_image)
                aug_image = self.rand_fill(aug_image)
                aug_image = self.noise(aug_image, p=0.2)

                aug_file = filename.replace('.png', '_augmented_{}_{}.png'.format(i, j))
                aug_image.save(aug_file)
                yield aug_file

    def process(self, node):
        '''
        Augment an airfoil plot given in a neo4j node

        :param node: The neo4j node data
        '''
        for filename in self.augment(node.data['filename']):
            yield self.default_transaction(data=dict(filename=filename, parent=node.data['parent']))
