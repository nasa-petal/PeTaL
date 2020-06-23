from .log import get_path

from cProfile import Profile as ProfileBase
from pstats   import Stats

import pickle

PROFILE_DIR = 'data/profiles/'

'''
Automatically profile bitflow modules
'''

class Profile:
    def __init__(self, name, directory=None):
        self.base = ProfileBase()
        self.base.enable()
        self.path = get_path(PROFILE_DIR, name, directory=directory, ending='.profile')

    def close(self):
        self.base.disable()
        stats  = Stats(self.base)
        stats.dump_stats(self.path)
