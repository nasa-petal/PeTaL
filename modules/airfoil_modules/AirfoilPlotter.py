from petal.pipeline.module_utils.module import Module

import pickle

import matplotlib.pyplot as plt

DPI  = 100

class AirfoilPlotter(Module):
    '''
    Plot a mined airfoil
    '''
    def __init__(self, name='AirfoilPlotter'):
        Module.__init__(self, in_label='Airfoil', out_label='CleanAirfoilPlot', connect_labels=('image', 'image'), name=name)

    def process(self, node, driver=None):
        coord_file  = node.data['coord_file']
        with open(coord_file, 'rb') as infile:
            coordinates = pickle.load(infile)
        fx, fy, sx, sy, camber = coordinates
        figsize  = (800/DPI, 200/DPI)
        plt.figure(figsize=figsize, dpi=DPI)
        plt.plot(fx, fy, color='black')
        plt.plot(sx, sy, color='black')
        plt.plot([sx[0], fx[0]], [sy[0], fy[0]], color='black') # Connect front
        plt.plot([sx[-1], fx[-1]], [sy[-1], fy[-1]], color='black') # Connect back
        plt.axis('off')
        filename = 'data/images/' + node.data['name'] + '_' + node.data['Ncrit'] + '_' + node.data['mach'] + node.data['Re'] + '.png'
        plt.savefig(filename)
        yield self.default_transaction(data=dict(filename=filename, parent=str(node.uuid)))


