from multiprocessing import Process, Queue
from collections import defaultdict
from importlib import import_module
from time import sleep
from uuid import uuid4

from driver import driver_listener
from batch import Batch

def batch_serializer(serialize_queue, transaction_queue, schedule_queue, sizes):
    batches = dict()
    i = 0
    while True:
        transaction = serialize_queue.get()
        label = transaction.out_label
        if label not in batches:
            batches[label] = Batch()
        batch = batches[label]
        batch.add(transaction)
        max_length = sizes.get(label, sizes['__default__'])
        if len(batch) >= max_length:
            filename = 'data/batches/{}'.format(uuid4())
            batch.save(filename)
            batch.clear()
            transaction_queue.put(filename)
            schedule_queue.put((label, filename))
        i += 1

def module_runner(module_name, serialize_queue, batch_file):
    module = import_module('modules.mining_modules.{}'.format(module_name))
    module = getattr(module, module_name)()
    
    i = 0
    for transaction in module.process():
        serialize_queue.put(transaction)
        i += 1

class Scheduler:
    def __init__(self, max_workers=10):
        self.transaction_queue = Queue(10000000)
        self.serialize_queue   = Queue(1000)
        self.schedule_queue    = Queue(100)
        self.driver_process    = Process(target=driver_listener,  args=(self.transaction_queue,))
        self.batch_process     = Process(target=batch_serializer, args=(self.serialize_queue, self.transaction_queue, self.schedule_queue, {'__default__': 100}))
        self.dependents        = defaultdict(list)
        self.workers           = []
        self.max_workers       = max_workers

    def schedule(self, module_name, batch_file=None):
        print('scheduled ', module_name, flush=True)
        self.workers.append(Process(target=module_runner, args=(module_name, self.serialize_queue, batch_file)))
        # TODO add to self.dependents

    def start(self):
        print('starting', flush=True)
        self.driver_process.start()
        self.batch_process.start()
        for process in self.workers:
            process.start()
        print('started', flush=True)

    def stop(self):
        print('stopped', flush=True)
        self.driver_process.terminate()
        self.batch_process.terminate()
        for process in self.workers:
            process.terminate()

    def check(self):
        print('checking..', flush=True)
        self.workers = [worker for worker in self.workers if worker.is_alive()]
        if len(self.workers) < self.max_workers:
            while not self.schedule_queue.empty():
                label, batch_file = self.schedule_queue.get()
                for sublabel in label.split(':'):
                    for dependent in self.dependents[sublabel]:
                        self.schedule(dependent, batch_file=batch_file)
