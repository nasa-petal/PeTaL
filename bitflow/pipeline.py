from time import time, sleep
import json
import os
import sys
import shutil

from .utils.utils import get_module_names, fetch

from .scheduler import Scheduler
from .utils.log import Log
from .utils.create_dependencies import create_dependencies


class BitflowInterface:
    '''
    This class defines an interface to a data mining server. 
    It allows modules and settings to the scheduler to be updated dynamically without stopping processing.
    For instance, one can run a species cataloger, and an image downloader that depends on it. 
    Then if the image downloader were to break mid-processing, it could be patched and would be re-loaded live, allowing the cataloger to continue running.
    '''
    def __init__(self, filename, module_dir='modules'):
        '''
        :param filename: A bitflow config file, JSON. See /config/ directory for examples
        :param module_dir: A directory with the relevant modules to be run
        '''
        self.module_dir = module_dir
        print('LOADING PeTaL config ({})'.format(filename), flush=True)
        create_dependencies(directory=module_dir)
        self.log = Log('bitflow_server')
        self.scheduler = Scheduler(filename, module_dir)
        self.times = dict()
        self.filename = filename
        self.sleep_time = 1
        self.reload_time = 30
        self.status_time = 1
        self.whitelist = []
        self.blacklist = []
        self.settings = self.load_settings()

    def reload_modules(self):
        '''
        Actively reload all modules
        '''
        for name in get_module_names(directory=self.module_dir):
            if len(self.whitelist) > 0:
                if name in self.whitelist:
                    self.scheduler.schedule(name)
            elif name not in self.blacklist:
                self.scheduler.schedule(name)

    def load_settings(self):
        '''
        Copy the settings file into the bitflows settings.
        For instance, can change the reload time from 30 to 10s if desired
        '''
        with open(self.filename, 'r') as infile:
            settings = json.load(infile)
        self.log.log(settings)
        for k, v in settings.items():
            if k.startswith('scheduler:'):
                k = k.replace('scheduler:', '')
                setattr(self.scheduler, k, v)
            elif k.startswith('bitflow:'):
                k = k.replace('bitflow:', '')
                setattr(self, k, v)
        return settings

    def start_server(self, clean=True):
        '''
        Start the bitflow server..
        :param clean: Remove data from a previous run of the server, i.e. batches and so on
        '''
        if clean:
            print('CLEANING Old Data', flush=True)
            self.clean()
        print('STARTING PeTaL Data Bitflow Server', flush=True)
        self.log.log('Starting bitflow server')
        start = time()
        self.reload_modules() 
        self.log.log('Starting scheduler')
        self.scheduler.start()
        done = False
        try:
            while not done:
                done = self.scheduler.check()
                sleep(self.sleep_time)
                duration = time() - start
                if duration > self.status_time:
                    self.scheduler.status(duration)
                if duration > self.reload_time:
                    start = time()
                    self.settings = self.load_settings()
                    self.reload_modules()
                    self.log.log('Actively reloading settings')
        except KeyboardInterrupt as interrupt:
            print('INTERRUPTING PeTaL Data Bitflow Server', flush=True)
        finally:
            print('STOPPING PeTaL Data Bitflow Server', flush=True)
            self.scheduler.stop()

    def clean(self):
        clean_dirs = ['logs', 'profiles', 'batches', 'images']
        for directory in ['logs', 'profiles', 'logs/modules', 'logs/paging', 'profiles/modules']:
            fulldir = 'data/' + directory
            try:
                shutil.rmtree(fulldir)
            except FileNotFoundError:
                pass
            os.mkdir(fulldir)
            with open(fulldir + '/.placeholder', 'w') as outfile:
                outfile.write('')
