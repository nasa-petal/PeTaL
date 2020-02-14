from time import time, sleep
from pprint import pprint
import json
import inspect
import os

from importlib import reload
from scheduler.scheduler import Scheduler
from scheduler.module_info import ModuleInfo

import modules


class PipelineInterface:
    '''
    This class defines an interface to a data mining server. It allows modules and settings to the scheduler to be updated dynamically without stopping processing.
    '''
    def __init__(self):
        self.scheduler = Scheduler()
        self.times = dict()
        self.load_settings()
        self.reload_modules()

    def reload_modules(self):
        print('Reloading modules', flush=True)
        global modules
        modules = reload(modules)
        for name, item in inspect.getmembers(modules):
            if inspect.isclass(item):
                filename = 'modules/mining_modules/{}.py'.format(name)
                if not os.path.isfile(filename):
                    filename = 'modules/machine_learning_modules/{}.py'.format(name)
                filetime = os.stat(filename).st_mtime
                if name not in self.times or self.times[name] != filetime:
                    self.times[name] = filetime
                    if name not in self.blacklist:
                        print('Reloading module: ', name)
                        self.scheduler.schedule(item())

    def load_settings(self, filename='settings.json'):
        with open('settings.json', 'r') as infile:
            settings = json.load(infile)
        pprint(settings)
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
                self.scheduler.check_added()
                sleep(self.sleep_time)
                self.scheduler.display()
                duration = time() - start
                if duration > self.reload_time:
                    start = time()
                    self.load_settings()
                    self.reload_modules()
        finally:
            print('Caught outer level exception, STOPPING server!')
            self.scheduler.stop()

if __name__ == '__main__':
    interface = PipelineInterface()
    interface.start_server()
