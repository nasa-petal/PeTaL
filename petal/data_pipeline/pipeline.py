from time import time, sleep
from pprint import pprint
import json
import os
import sys

from scheduler import Scheduler

# import modules

class PipelineInterface:
    '''
    This class defines an interface to a data mining server. It allows modules and settings to the scheduler to be updated dynamically without stopping processing.
    '''
    def __init__(self, filename):
        self.scheduler = Scheduler()
        self.times = dict()
        self.filename = filename
        self.sleep_time = 1
        self.reload_time = 30
        self.whitelist = []
        self.blacklist = []
        self.load_settings()

    def reload_modules(self):
        mining_modules = os.listdir('modules/mining_modules/')
        ml_modules     = os.listdir('modules/machine_learning_modules/')
        modules = mining_modules + ml_modules
        for filename in modules:
            if filename.endswith('.py'):
                name = os.path.basename(filename).split('.')[0]
                if len(self.whitelist) > 0:
                    if name in self.whitelist:
                        self.scheduler.schedule(name)
                elif name not in self.blacklist:
                    self.scheduler.schedule(name)

    def load_settings(self):
        with open(self.filename, 'r') as infile:
            settings = json.load(infile)
        # pprint(settings)
        for k, v in settings.items():
            if k.startswith('scheduler:'):
                k = k.replace('scheduler:', '')
                setattr(self.scheduler, k, v)
            elif k.startswith('pipeline:'):
                k = k.replace('pipeline:', '')
                setattr(self, k, v)

    def start_server(self):
        print('Starting pipeline server', flush=True)
        start = time()
        self.reload_modules() 
        print('Starting scheduler', flush=True)
        self.scheduler.start()
        try:
            while True:
                self.scheduler.check()
                sleep(self.sleep_time)
                duration = time() - start
                if duration > self.reload_time:
                    start = time()
                    self.load_settings()
                    self.reload_modules()
        finally:
            print('Caught outer level exception, STOPPING server!')
            self.scheduler.stop()

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        settings_file = 'settings.json'
    else:
        settings_file = args[0]
    interface = PipelineInterface(settings_file)
    interface.start_server()
