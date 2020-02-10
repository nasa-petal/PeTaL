from time import time

class ModuleInfo:
    def __init__(self, total):
        self.start = time()
        self.current = 0
        self.total   = total

    def set_current(self, c):
        self.current = c

    def get_current(self):
        return self.current

    def check_info(self, current):
        self.current = current
        info = dict()
        info['total_time']  = time() - self.start
        info['percent']     = current / self.total * 100.0
        return info

    def __str__(self):
        info = self.check_info(self.current)
        return '{percent:5.2f}'.format(**info)

