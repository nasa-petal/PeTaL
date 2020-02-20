from driver import driver_listener

from multiprocessing import Process, Queue

from time import sleep

from importlib import import_module

def module_runner(module_name, transaction_queue, batch_queue):
    module = import_module('modules.mining_modules.{}'.format(module_name))
    module = getattr(module, module_name)()

    print(module_name, flush=True)
    print(module)

class Scheduler:
    def __init__(self):
        self.transaction_queue = Queue()
        self.batch_queue       = Queue()
        self.driver_process    = Process(target=driver_listener, args=(self.transaction_queue, self.batch_queue))
        self.workers = []

    def schedule(self, module_name):
        print('scheduled ', module_name, flush=True)
        self.workers.append(Process(target=module_runner, args=(module_name, self.transaction_queue, self.batch_queue)))

    def start(self):
        print('starting', flush=True)
        self.driver_process.start()
        for process in self.workers:
            process.start()
        print('started', flush=True)

    def stop(self):
        print('stopped', flush=True)
        self.driver_process.terminate()
        for process in self.workers:
            process.terminate()

    def check(self):
        print('checking..', flush=True)
