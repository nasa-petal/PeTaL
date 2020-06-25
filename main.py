#!/usr/bin/env python
from bitflow import BitflowInterface
import sys

'''
Create an interface to the PeTaL data pipeline and tell it where to import modules from
'''

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        settings_file = 'config/default.json'
    else:
        settings_file = args[0]
    interface = BitflowInterface(settings_file, module_dir='modules')
    interface.log.log('Loaded settings from ', settings_file)
    interface.start_server(clean=True)
