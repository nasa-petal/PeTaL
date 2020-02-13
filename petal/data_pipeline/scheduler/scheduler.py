from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from collections import defaultdict, namedtuple
from uuid import uuid4
from time import sleep

from .driver import Driver
from .label_tracker import LabelTracker
from .module_info import ModuleInfo

UPPER_BOUND = 90000000000 # Allow up to 90 billion nodes to accumulate if they have no dependent consumers

class ModuleProcess():
    def __init__(self, process, module, info):
        self.process = process
        self.module = module
        self.info = info

driver = Driver(1000, 0.25)

def driver_runner(module, tracker, info, ids):
    for i, node_id in enumerate(ids):
        node_id = str(node_id)
        info.set_current(i)
        driver.run_id(module, tracker, node_id)

def driver_independent_runner(module, tracker, info):
    driver.run(module, tracker, info)

def driver_page_runner(module, tracker, info):
    driver.run_page(module, tracker, info)

class Scheduler:
    '''
    This class dynamically schedules modules in batches, and provides a solution to the producer-consumer problem within the data pipeline
    '''
    def __init__(self, accumulate_limit=5, max_running=20):
        self.accumulate_limit = accumulate_limit

        BaseManager.register('LabelTracker', LabelTracker)
        BaseManager.register('ModuleInfo', ModuleInfo)
        self.manager = BaseManager()
        self.manager.start()
        self.label_tracker = self.manager.LabelTracker()
        self.label_counts = dict()

        self.dependents = defaultdict(list)
        self.queue      = []
        self.running    = []
        self.finished   = set()

        self.max_running = max_running

    def init(self, f, module, args=None, count=None):
        if args is None:
            args = ()
        if count is None:
            count = module.count
        info = self.manager.ModuleInfo(count)
        self.queue.append(ModuleProcess(Process(target=f, args=(module, self.label_tracker, info) + args), module, info))

    def schedule(self, module):
        independent = module.in_label is None
        if independent:
            self.init(driver_independent_runner, module)
        else:
            self.dependents[module.in_label].append(module)
            self.init(driver_page_runner, module)
            print('Added paged runner', flush=True)

    def start(self):
        for p in self.queue:
             p.process.start()
        self.running = self.queue
        self.queue = []

    def check_added(self):
        for label, id_set in self.label_tracker.get().items():
            if label in self.dependents:
                schedule_dependent = (len(id_set) > self.accumulate_limit or label in self.finished) and self.max_running - len(self.running) > 0
                schedule_dependent = schedule_dependent and label not in {mp.module.in_label for mp in self.running}
                if label in self.finished:
                    self.finished.remove(label)
                if schedule_dependent:
                    dep_modules = self.dependents[label]
                    ids = list(id_set)
                    for module in dep_modules:
                        print('Scheduled ', module, ' for {} nodes'.format(len(ids)), flush=True)
                        self.init(driver_runner, module, args=(ids,), count=len(ids))
                        self.label_tracker.set_throttle_count(label, self.accumulate_limit)
                        if label in self.label_counts:
                            self.label_counts[label] += len(ids)
                        else:
                            self.label_counts[label] = len(ids)
                    self.label_tracker.clear(label)
            else:
                self.label_tracker.set_throttle_count(label, UPPER_BOUND)


    def display(self):
        to_start = min(self.max_running - len(self.running), len(self.queue))
        if to_start > 0:
            print(len(self.running), ' processes are running', flush=True)
            print('Starting ', to_start, ' processes', flush=True)
        for i in range(to_start):
            p = self.queue[i]
            p.process.start()
            self.running.append(p)
        self.queue = self.queue[to_start:]

        for k, v in self.label_counts.items():
            print('{:>10} : {:<10}'.format(k, v), flush=True)


        for p in self.running:
            if not p.process.is_alive():
                print('Finished: ', p.module)
                self.finished.add(p.module.out_label)
            else:
                # if p.info.get_current() > 0:
                # print(p.module, flush=True)
                # print(str(p.info), flush=True)
                pass
        self.running = [p for p in self.running if p.process.is_alive()]

    def stop(self):
        for p in self.running:
            p.process.terminate()

