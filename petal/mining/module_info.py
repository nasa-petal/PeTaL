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
        self.process = psutil.Process(os.getpid())

    def set_current(self, c):
        self.current = c

    def get_current(self):
        return self.current

    def check_info(self):
        info = dict()
        info['total_time']  = time() - self.start
        info['percent']     = self.current / self.total * 100.0
        info['current'] = self.current
        # info['mem_percent'] = self.process.mem_percent()
        # info['cpu_percent'] = self.process.cpu_percent()
        info['rate'] = self.current / info['total_time']
        return info

    def __str__(self):
        info = self.check_info()
        return '{percent:10.5f}% done at {rate:5.2f} calls/second'.format(percent=info['percent'], rate=info['rate'])

    def __repr__(self):
        return str(self)

