from driver import driver_listener

from multiprocessing import Process, Queue

from time import sleep

from importlib import import_module

from batch import Batch

from uuid import uuid4

def batch_serializer(serialize_queue, batch_queue, sizes):
    batches = dict()
    i = 0
    while True:
        label, uuid = serialize_queue.get()
        if label not in batches:
            batches[label] = Batch()
        batch = batches[label]
        batch.add(uuid)
        max_length = sizes.get(label, sizes['__default__'])
        if len(batch) >= max_length:
            filename = 'data/batches/{}'.format(uuid4())
            batch.save(filename)
            batch.clear()
            print('saved batch', flush=True)
            batch_queue.put(filename)
        if i % 100 == 0:
            print('batcher: ', i, flush=True)
        i += 1
    print('done', flush=True)

def module_runner(module_name, transaction_queue, serialize_queue):
    module = import_module('modules.mining_modules.{}'.format(module_name))
    module = getattr(module, module_name)()
    
    i = 0
    for transaction in module.process():
        if i % 100000 == 0:
            print('cataloger: ', i, flush=True)
        serialize_queue.put((transaction.out_label, transaction.uuid))
        transaction_queue.put(transaction)
        i += 1

class Scheduler:
    def __init__(self):
        self.transaction_queue = Queue()
        self.batch_queue       = Queue()
        self.serialize_queue   = Queue()
        self.driver_process    = Process(target=driver_listener,  args=(self.transaction_queue, self.serialize_queue))
        self.batch_process     = Process(target=batch_serializer, args=(self.serialize_queue, self.batch_queue, {'__default__': 100}))
        self.workers = []

    def schedule(self, module_name):
        print('scheduled ', module_name, flush=True)
        self.workers.append(Process(target=module_runner, args=(module_name, self.transaction_queue, self.serialize_queue)))

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
        # self.workers = [worker for worker in self.workers if worker.is_alive()]
