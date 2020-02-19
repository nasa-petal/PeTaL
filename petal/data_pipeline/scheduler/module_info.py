from time import time
import os
# import psutil, os

class ModuleInfo:
    '''
    Used for self-benchmarking individual modules
    '''
    def __init__(self, total):
        self.start = time()
        self.current = 0
        self.total   = total
        self.cpu_percent = 0.0
        self.mem_percent = 0.0

    def set(self, b):
        b_start, b_current, b_cpu_percent, b_mem_percent = b.get()
        self.start       = min(self.start, b_start)
        self.current     = self.current + b_current
        self.cpu_percent = self.cpu_percent + b_cpu_percent
        self.mem_percent = self.mem_percent + b_mem_percent

    def get(self):
        return self.start, self.current, self.cpu_percent, self.mem_percent

    def add(self, b):
        x = ModuleInfo(self.total)
        x.set(b)
        return x
    
    def set_usage(self, mem, cpu):
        self.mem_percent = mem
        self.cpu_percent = cpu

    def set_current(self, c):
        self.current = c

    def get_current(self):
        return self.current

    def check_info(self):
        info = dict()
        info['total_time']  = time() - self.start
        info['percent']     = self.current / self.total * 100.0
        info['current'] = self.current
        info['mem_percent'] = self.mem_percent
        info['cpu_percent'] = self.cpu_percent
        info['rate'] = self.current / info['total_time'] if info['total_time'] > 0 else 0.0
        return info

    def __str__(self):
        info = self.check_info()
        return 'rate: {rate:7.4f}, total: {total:10}'.format(rate=info['rate'], total=info['current'])

    def __repr__(self):
        return str(self)

