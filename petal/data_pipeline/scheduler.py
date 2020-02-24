from multiprocessing import Process, Queue, Pool
from queue import Empty
from collections import defaultdict
from importlib import import_module
from time import sleep, time
from uuid import uuid4

from driver import driver_listener
from batch import Batch

def batch_serializer(serialize_queue, transaction_queue, schedule_queue, sizes):
    start = time()
    batches = dict()
    i = 0
    while True:
        try:
            transaction = serialize_queue.get(block=False)
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
        except Empty:
            for label, batch in batches.items():
                if len(batch) > 0:
                    filename = 'data/batches/{}'.format(uuid4())
                    batch.save(filename)
                    batch.clear()
                    transaction_queue.put(filename)
                    schedule_queue.put((label, filename))
        duration = time() - start
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
        gen = (transaction for item in batch.items for transaction in module.process(item))
    i = 0
    for transaction in gen:
        serialize_queue.put(transaction)
        i += 1

class Scheduler:
    def __init__(self, max_workers=30, n_drivers=1):
        self.transaction_queue = Queue(10000)
        self.serialize_queue   = Queue(10000)
        self.schedule_queue    = Queue(1000)
        self.driver_processes  = [Process(target=driver_listener,  args=(self.transaction_queue,)) for i in range(n_drivers)]
        self.batch_process     = Process(target=batch_serializer, args=(self.serialize_queue, self.transaction_queue, self.schedule_queue, {'__default__': 10}))
        self.dependents        = defaultdict(list)
        self.workers           = []
        self.waiting           = []
        self.max_workers       = max_workers
        self.dependencies      = dict()

    def get_type_signature(self, module_name):
        if module_name not in self.dependencies:
            return None, None
        depends = self.dependencies[module_name]
        in_label, out_label = depends["in"], depends["out"]
        if in_label == 'None':
            in_label = None
        if out_label == 'None':
            out_label = None
        return in_label, out_label

    def schedule(self, module_name):
        in_label, out_label = self.get_type_signature(module_name)
        print(module_name, in_label, out_label, flush=True)
        if in_label is None:
            print('Starting ', module_name, flush=True)
            self.workers.append((module_name, Process(target=module_runner, args=(module_name, self.serialize_queue, None))))
        else:
            self.dependents[in_label].append(module_name)

    def start(self):
        for process in self.driver_processes:
            process.start()
        self.batch_process.start()
        for name, process in self.workers:
            process.start()

    def stop(self):
        for process in self.driver_processes:
            process.terminate()
        self.batch_process.terminate()
        for name, process in self.workers:
            process.terminate()

    def add_proc(self, dep_proc):
        dependent, process = dep_proc
        print('Starting dependent', dependent, flush=True)
        self.workers.append((dependent, process))
        self.workers[-1][1].start()

    def check(self):
        print('checking.. ', flush=True)
        self.workers = [(name, worker) for name, worker in self.workers if worker.is_alive()]
        while len(self.waiting) > 0:
            if len(self.workers) < self.max_workers:
                self.add_proc(self.waiting.pop())
            else:
                break
        while not self.schedule_queue.empty():
            if len(self.workers) < self.max_workers:
                label, batch_file = self.schedule_queue.get(block=False)
                for sublabel in label.split(':'):
                    for dependent in self.dependents[sublabel]:
                        dep_proc = (dependent, Process(target=module_runner, args=(dependent, self.serialize_queue, batch_file)))
                        if len(self.workers) < self.max_workers:
                            self.add_proc(dep_proc)
                        else:
                            self.waiting.append(dep_proc)
            else:
                break
        print([t[0] for t in self.workers])
        print(len(self.workers) == 0, self.schedule_queue.empty(), self.serialize_queue.empty(), self.transaction_queue.empty(), flush=True)
        print(len(self.workers),      self.schedule_queue.qsize(), self.serialize_queue.qsize(), self.transaction_queue.qsize(), flush=True)
        if len(self.workers) == 0 and self.serialize_queue.empty() and self.transaction_queue.empty():
            return True
        return False
