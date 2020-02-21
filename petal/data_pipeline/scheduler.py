from multiprocessing import Process, Queue, Pool
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

def fetch(module_name):
    try:
        module = import_module('modules.mining_modules.{}'.format(module_name))
    except:
        module = import_module('modules.machine_learning_modules.{}'.format(module_name))
    return getattr(module, module_name)()

def module_runner(module_name, serialize_queue, batch_file):
    module = fetch(module_name)
    
    if batch_file is None:
        gen = module.process()
    else:
        batch = Batch()
        batch.load(batch_file)
        gen = (transaction for item in batch.items for transaction in module.process(item.data))
    i = 0
    for transaction in gen:
        serialize_queue.put(transaction)
        i += 1

class Scheduler:
    def __init__(self, max_workers=10):
        self.transaction_queue = Queue()
        self.serialize_queue   = Queue(100000)
        self.schedule_queue    = Queue()
        self.driver_process    = Process(target=driver_listener,  args=(self.transaction_queue,))
        self.batch_process     = Process(target=batch_serializer, args=(self.serialize_queue, self.transaction_queue, self.schedule_queue, {'__default__': 10000}))
        self.dependents        = defaultdict(list)
        self.workers           = []
        self.max_workers       = max_workers
        self.dependencies      = dict()

    def get_type_signature(self, module_name):
        module = fetch(module_name)
        return module.in_label, module.out_label
        # if module_name not in self.dependencies:
        #     return None, None
        # depends = self.dependencies[module_name]
        # in_label, out_label = depends["in"], depends["out"]
        # return in_label, out_label

    def schedule(self, module_name):
        in_label, out_label = self.get_type_signature(module_name)
        if in_label is None:
            self.workers.append(Process(target=module_runner, args=(module_name, self.serialize_queue, None)))
        else:
            self.dependents[in_label].append(module_name)

    def start(self):
        self.driver_process.start()
        self.batch_process.start()
        for process in self.workers:
            process.start()

    def stop(self):
        self.driver_process.terminate()
        self.batch_process.terminate()
        for process in self.workers:
            process.terminate()

    def check(self):
        self.workers = [worker for worker in self.workers if worker.is_alive()]
        while not self.schedule_queue.empty():
            if len(self.workers) < self.max_workers:
                label, batch_file = self.schedule_queue.get()
                for sublabel in label.split(':'):
                    for dependent in self.dependents[sublabel]:
                        print('Starting dependent', dependent, flush=True)
                        self.workers.append(Process(target=module_runner, args=(dependent, self.serialize_queue, batch_file)))
                        self.workers[-1].start()
