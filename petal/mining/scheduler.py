from uuid import uuid4
from time import sleep

from multiprocessing import Process

from driver import Driver

driver = Driver(page_size=10, rate_limit=0.2)

def driver_runner(module):
    driver.run(module)

class Scheduler:
    def __init__(self, accumulate_limit=100, page_size=1, rate_limit=0.25):
        # self.driver   = Driver(page_size=page_size, rate_limit=rate_limit)
        self.queue    = []
        self.running  = []
        self.affected = dict() # label, module-to-schedule

    def schedule(self, module):
        self.queue.append(Process(target=driver_runner, args=(module,)))

    def start(self):
        for process in self.queue:
            process.start()

    def stop(self):
        for process in self.queue:
            process.terminate()

from modules import WikipediaModule, BackboneModule, EOLModule, GoogleScholarModule, HighwireModule, JEBModule

if __name__ == '__main__':
    scheduler = Scheduler()
    try:
        scheduler.schedule(WikipediaModule())
        scheduler.schedule(EOLModule())
        scheduler.schedule(JEBModule())

        scheduler.start()
        sleep(100)
    finally:
        scheduler.stop()

    # backbone        = BackboneModule()
    # highwire        = HighwireModule()
    # scholar_scraper = GoogleScholarModule()

