from time import time
import psutil, os

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

    def add(self, b):
        return self
        # x = ModuleInfo(self.total)
        # x.start       = min(self.start, b.start)
        # x.current     = self.current + b.current
        # x.cpu_percent = self.cpu_percent + b.cpu_percent
        # x.mem_percent = self.mem_percent + b.mem_percent
        # return x
    
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
        return '{percent:10.5f}% done, rate: {rate:7.4f}'.format(percent=info['percent'], rate=info['rate'])
        # return '{percent:10.5f}% done, rate: {rate:7.4f}, mem: {memory:7.4f}, cpu: {cpu:7.4f},'.format(percent=info['percent'], rate=info['rate'], memory=info['mem_percent'], cpu=info['cpu_percent'])

    def __repr__(self):
        return str(self)

