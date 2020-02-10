from uuid import uuid4
from time import sleep

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager

from collections import defaultdict

from driver import Driver

driver = Driver(page_size=1, rate_limit=0.2)

def driver_runner(module, tracker, ids):
    driver.run_page(module, tracker, ids)

def driver_independent_runner(module, tracker):
    driver.run(module, tracker)

class LabelTracker():
    def __init__(self):
        self.tracker = dict()

    def add(self, label, uuid):
        if label in self.tracker:
            self.tracker[label].add(uuid)
        else:
            self.tracker[label] = {uuid}

    def get(self):
        return self.tracker

    def clear(self, label):
        self.tracker[label].clear()

class Scheduler:
    def __init__(self, accumulate_limit=3):
        self.accumulate_limit = accumulate_limit

        BaseManager.register('LabelTracker', LabelTracker)
        self.manager = BaseManager()
        self.manager.start()
        self.label_tracker = self.manager.LabelTracker()

        self.dependents = defaultdict(list)
        self.queue      = []
        self.finished   = set()

    def schedule(self, module):
        independent = module.in_label is None
        if independent:
            self.queue.append((Process(target=driver_independent_runner, args=(module, self.label_tracker)), module))
        else:
            self.dependents[module.in_label].append(module)

    def start(self):
        for process, _ in self.queue:
            process.start()

    def check_added(self):
        for label, id_set in self.label_tracker.get().items():
            schedule_dependent = len(id_set) > self.accumulate_limit or label in self.finished
            if label in self.finished:
                self.finished.remove(label)
            if schedule_dependent:
                dep_modules = self.dependents[label]
                ids = list(id_set)
                for module in dep_modules:
                    print('Scheduled dependent module ', module, ' on ', label, ' for {} nodes'.format(len(ids)), flush=True)
                    self.queue.append((Process(target=driver_runner, args=(module, self.label_tracker, ids)), module))
                    self.queue[-1][0].start()
                self.label_tracker.clear(label)

    def display(self):
        if len(self.queue) == 0:
            print('Done')
            return True
        for process, module in self.queue:
            if process.is_alive():
                # print('Currently running: ', module)
                pass
            else:
                print('Finished: ', module)
                self.finished.add(module.out_label)
        self.queue = [(p, m) for p, m in self.queue if p.is_alive()]
        return False

    def stop(self):
        for process, _ in self.queue:
            process.terminate()

from modules import WikipediaModule, BackboneModule, EOLModule, GoogleScholarModule, HighwireModule, JEBModule

if __name__ == '__main__':
    scheduler = Scheduler()
    try:
        scheduler.schedule(BackboneModule())
        scheduler.schedule(WikipediaModule())
        # scheduler.schedule(EOLModule())
        # scheduler.schedule(JEBModule())

        scheduler.start()
        done = False
        while not done:
            scheduler.check_added()
            sleep(.01)
            done = scheduler.display()
    finally:
        scheduler.stop()

    # highwire        = HighwireModule()
    # scholar_scraper = GoogleScholarModule()

