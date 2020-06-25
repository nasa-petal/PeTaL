from multiprocessing import Process, Queue, Pool, Manager
from queue import Empty
from collections import defaultdict
from collections import Counter
from time import sleep, time

import json

from .utils.utils import fetch
from .batch import Batch
from .driver import Driver, driver_listener
from .utils.log import Log

'''
This is definitely the most complicated code of the data bitflow, but justifiably so.
The scheduler manages running processes and, unsurprisingly, schedules new ones.
It separates data streams, such as database transactions, which allows bitflow modules to run independently.
There are three streams in total:
    serialize_queue: data is scheduled to be written to disk (separates disk-writing from cpu-bound processing)
    schedule_queue: data is given to modules which will run using it (i.e. species names to download articles for)
    transaction_queue: data is scheduled to be written to the neo4j database (slow for some reason, usually because neo4j likes bulk data. See the neo4j LOAD CSV command for exceptionally fast data handling)

This code is written this way because I believe it to be faster after having profiled it quite a few times.
However, there is definitely large potential for simplification or further optimization.
See data/profiles for relevant performance characteristics
'''

FORCE_SAVE_INTERVAL = 10 # Forcibly save batches every N seconds, even if they are smaller than required. Helpful if consuming processes are faster than producing processes

def save_batch(schedule_queue, transaction_queue, batch):
    '''
    Save a batch to file and split it into a queue for scheduling new processes, and a queue for adding transactions to the database 

    :param schedule_queue: The input stream of new data for new processes
    :param transaction_queue: The input stream of transactions to be entered permanently in the database
    '''
    batch.save()
    transaction_queue.put(batch)
    schedule_queue.put(batch)

from collections import defaultdict

def batch_serializer(serialize_queue, transaction_queue, schedule_queue, sizes):
    '''
    Save incoming data to disk, and send it out for further processing
    Breaks data into batches defined by the sizes in the sizes dictionary
    Data comes in from serialize_queue and is split into transaction_queue and schedule_queue.

    :param serialize_queue: The stream of data to be written to disk
    :param schedule_queue: The input stream of new data for new processes
    :param transaction_queue: The input stream of transactions to be entered permanently in the database
    :param sizes: (dict) Limits on the batch-size of particular neo4j labels
    '''
    batches   = dict()
    counts    = defaultdict(lambda : 0)
    durations = dict()
    while True:
        try:
            transaction = serialize_queue.get(block=False)
            label = transaction.out_label
            if label is not None:
                for sublabel in label.split(':'):
                    # Start a new batch if necessary
                    if sublabel not in batches:
                        batches[sublabel] = Batch(sublabel, uuid=str(sublabel) + '_' + str(counts[sublabel]))
                        durations[sublabel] = time()
                    batch = batches[sublabel]
                    # Add to the batch until it is long enough
                    batch.add(transaction)
                    max_length = sizes.get(sublabel, sizes['__default__'])
                    if len(batch) >= max_length:
                        save_batch(schedule_queue, transaction_queue, batches.pop(sublabel))
                        counts[sublabel]  += 1
        except Empty: # If incoming data is slow, forcibly save batches
            labels = list(batches.keys())
            for label in labels:
                if time() - durations[label] > FORCE_SAVE_INTERVAL:
                    save_batch(schedule_queue, transaction_queue, batches.pop(label))
                    counts[label]  += 1
                    durations[label] = time()

def run_module(module, serialize_queue, batch):
    '''
    Helper function which simply runs a PeTaL module on data from batch and puts the result in serialize_queue.

    :param module: An instantiated PeTaL module, i.e. `WikipediaArticleModule()`
    :param serialize_queue: The stream of data to be written to disk, and later processed
    :param batch: The data on which this module depends
    '''
    module.log.log('Initiated ', module.name, ' run_module() in scheduler')
    # Handle "independent" modules (i.e. the species cataloger which has type None -> Taxon)
    if batch is None:
        module.log.log('Backbone run ', module.name)
        gen = module.process()
    # Handle "dependent" modules (i.e. the article downloader which has type Taxon -> Article)
    else:
        module.log.log('Batched returning run with ', module.name)
        gen = module.process_batch(batch)
    # If new data has been generated, queue it for further processing
    if gen is not None:
        for transaction in gen:
            module.log.log(transaction)
            serialize_queue.put(transaction)
    module.log.log('Finished queueing transactions from ', module.name)

def module_runner(module_name, serialize_queue, batch, settings_file, module_dir='modules'):
    '''
    A helper function for running a module by name.
    This function is scheduled by the scheduler, either when the module is independent, or is dependent and has input data

    :param module_name: The name of the class, i.e. 'WikipediaArticleModule'
    :param serialize_queue: The stream of data to be written to disk, and later processed
    :param batch: A batch of data for dependent processes
    :param settings_file: filename of JSON file with Driver settings
    :param module_dir: Directory containing modules subfolders
    '''
    with fetch(module_name, directory=module_dir) as module:
        module.add_driver(Driver(settings_file))
        run_module(module, serialize_queue, batch)

def pager(name, label, serialize_queue, settings_file, delay, page_size, module_dir='modules'):
    '''
    Continually running "pager", which feeds data from the neo4j database into a process.
    Toggled by derived settings within a `Module` class.

    :param name: Name of the Module being fed paged data
    :param label: Neo4j node label to fetch data for
    :param serialize_queue: The stream of data to be written to disk, and later processed
    :param delay: Time in seconds to sleep for
    :param page_size: Fetch this many batches at a time. 
    :param module_dir: Where the modules live...
    '''
    log = Log(name=name, directory='paging')

    driver = Driver(settings_file) # neo4j connection to get data from

    module = fetch(name, directory=module_dir)
    module.add_driver(driver)

    batch_counts = Counter()
    matcher = 'MATCH (n:Batch) WHERE n.label = \'{}\' '.format(label)

    while True:
        query = matcher + 'WITH COUNT (n) AS count RETURN count'
        log.log('Paging using query: ', query)
        # Count Batches with this label time. Helpful if ':Batch(label)' is an index within the database
        # Use CREATE INDEX ON :Batch(label) if it is not
        # TODO The above, automatically
        count = next(driver.run_query(query).records())['count']
        log.log(name, ' page count: ', count)
        if count == 0:
            continue

        for i in range(count // page_size + 1): # Plus one accounts for items past a multiple of the page size, i.e. 11 items when page size is 10
            # Get the next batch of nodes
            page_query = matcher + 'RETURN (n) SKIP {} LIMIT {}'.format(i * page_size, page_size)
            log.log('Page query: ', page_query)
            pages = driver.run_query(page_query).records()
            for page in (node['n'] for node in pages):
                # Load a batch from neo4j
                label = page['label']
                uuid  = page['uuid']
                rand  = page['rand']
                # If the module has the 'epochs' field, feed batches multiple times
                # Otherwise, feed each batch once.
                max_count = module.epochs if hasattr(module, 'epochs') else 1
                if batch_counts[uuid] < max_count:
                    batch_counts[uuid] += 1
                    log.log('Running page: ', str(uuid))
                    batch = Batch(label, uuid=uuid, rand=rand)
                    batch.load()
                    run_module(module, serialize_queue, batch)
        sleep(delay)

class Scheduler:
    '''
    A complicated but essential class. Candidate for simplification.

    Simply schedules modules based on type-signature, and handles the flow of data to and from a neo4j database.
    '''
    def __init__(self, settings_file, module_dir):
        '''
        :param settings_file: JSON file containing modules to run, database info, batch sizes, and timing settings
        :param module_dir: Location of module subdirectories, i.e. /modules/ where modules are stored like /modules/scraping/Wikipedia.py
        '''
        self.log = Log('scheduler')

        # Start by loading various settings, importantly job and data limits
        self.module_dir = module_dir
        self.settings_file = settings_file
        with open(settings_file, 'r') as infile:
            self.settings = json.load(infile)
        self.max_workers = self.settings['scheduler:max_workers']
        self.sizes  = self.settings['batch_sizes']
        self.limits = self.settings['process_limits']

        # Initialize the three data streams used by the scheduler.
        # transaction_queue: data is scheduled to be written to the neo4j database 
        #    (slow for some reason, usually because neo4j likes bulk data. 
        #       See the neo4j LOAD CSV command for exceptionally fast data handling)
        self.transaction_queue = Queue()
        # serialize_queue: data is scheduled to be written to disk (separates disk-writing from cpu-bound processing)
        self.serialize_queue   = Queue()
        # schedule_queue: data is given to modules which will run using it (i.e. species names to download articles for)
        self.schedule_queue    = Queue()

        # Initialize process for putting data streams to disk (serializer_process)
        self.serializer_process = Process(target=batch_serializer, args=(self.serialize_queue, self.transaction_queue, self.schedule_queue, self.sizes))
        self.serializer_process.daemon = True
        # Initialize process for putting data streams to neo4j database (driver_process)
        self.driver_process = Process(target=driver_listener,  args=(self.transaction_queue, settings_file))
        self.driver_process.daemon = True


        # Track dependents (i.e. WikipediaModule as Taxon -> Article)
        self.dependents        = defaultdict(set)

        self.workers           = [] # The active working processes
        self.pagers            = [] # Processes that feed data from neo4j to running processes
        self.waiting           = [] # Processes waiting to be run

        with open('.dependencies.json', 'r') as infile:
            self.dependencies = json.load(infile)

    def schedule(self, module_name):
        '''
        Add a module, by name, to be run by the scheduler.
        If independent, it runs when the scheduler is started (or immediately if the scheduler is running)
        If dependent, it runs when appropriate data comes in, and the scheduler is running

        :param module_name: The name of the module (str), i.e. WikipediaArticleModule
        '''
        self.log.log('Scheduling ', module_name)
        # Get dependencies of the module
        try:
            in_label, out_label, page = self.dependencies[module_name]
        except KeyError:
            raise RuntimeError('The module ' + module_name + ' is not configured correctly. Check the module itself and .dependencies.json')

        # If the module expects data from neo4j, start a paging process to passively read data from neo4j
        if page:
            self.log.log('Starting pager for', module_name)
            proc = Process(target=pager, args=(module_name, in_label, self.serialize_queue, self.settings_file, self.settings['pager_delay'], self.settings['page_size'], self.module_dir))
            proc.daemon = True
            self.pagers.append(proc)
        # If the module is independent, add it to active workers
        elif in_label is None:
            proc = Process(target=module_runner, args=(module_name, self.serialize_queue, None, self.settings_file, self.module_dir))
            proc.daemon = True
            self.workers.append((module_name, proc))
        # If the module is dependent, track this dependency for when input data comes in
        else:
            self.add_dependents(in_label, module_name)

    def add_dependents(self, in_label, module_name):
        '''
        Add a module as a dependent, possibly for multiple labels

        :param in_label: The label to run this module on
        :param module_name: The module to run 
        '''
        if in_label is not None:
            for label in in_label.split(','):
                for sublabel in label.split(':'):
                    if module_name not in self.dependents[sublabel]:
                        self.log.log('Added dependent: ', module_name)
                        self.dependents[sublabel].add(module_name)

    def start(self):
        '''
        Simply start the scheduler.
        '''
        self.driver_process.start()
        self.serializer_process.start()
        for name, process in self.workers:
            print('  Starting ', name, flush=True)
            self.log.log('Starting ', name)
            process.start()
        for pager in self.pagers:
            pager.start()

    def stop(self):
        '''
        Simply stop the scheduler.
        '''
        self.driver_process.terminate()
        self.serializer_process.terminate()
        for name, process in self.workers:
            process.terminate()
        for pager in self.pagers:
            pager.terminate()

    def start_process(self, dep_proc):
        '''
        Simply start a new dependent process that has been waiting.

        :param dep_proc: Dependent process to start
        '''
        dependent, process = dep_proc
        self.log.log('Starting dependent ', dependent, ' ', process)
        process.start()
        print('  Started ', dependent, flush=True)
        self.workers.append((dependent, process))

    def check_limit(self, dependent):
        '''
        Check if the scheduler is running within worker limits, defined in the JSON settings file

        :param dependent: A module name related to a worker (process) limit
        '''
        count = 0
        for name, worker in self.workers:
            if name == dependent:
                count += 1
        upper = self.limits.get(dependent, self.limits['__default__'])
        return count < upper

    def check(self):
        '''
        Run checks on currently running processes, and potentially start more.
        '''
        # Remove finished processes
        self.workers = [(name, worker) for name, worker in self.workers if worker.is_alive()]
        # Start waiting processes
        while len(self.waiting) > 0:
            dependent, proc = self.waiting[-1]
            if len(self.workers) < self.max_workers and self.check_limit(dependent):
                self.start_process(self.waiting.pop())
            else:
                break
        # Schedule dependent processes
        while not self.schedule_queue.empty():
            if len(self.workers) < self.max_workers:
                batch = self.schedule_queue.get(block=False)
                if batch.label is not None:
                    for sublabel in batch.label.split(':'):
                        for dependent in self.dependents[sublabel]:
                            proc = Process(target=module_runner, args=(dependent, self.serialize_queue, batch, self.settings_file, self.module_dir))
                            proc.daemon = True
                            dep_proc = (dependent, proc)
                            if len(self.workers) < self.max_workers and self.check_limit(dependent):
                                self.start_process(dep_proc)
                            else:
                                self.waiting.append(dep_proc)
            else:
                break
        # Optionally can return true if the server should be stopped.
        return False

    def status(self, duration):
        '''
        Report the schedulers status

        :param duration: Time in seconds the scheduler has run?
        '''
        running = dict()
        for dep, _ in self.workers:
            if dep in running:
                running[dep] += 1
            else:
                running[dep] = 1
        waiting_counts = dict()
        for dep, _ in self.waiting:
            if dep in waiting_counts:
                waiting_counts[dep] += 1
            else:
                waiting_counts[dep] = 1

        running_str = ' '.join('{} ({})'.format(dep, count) for dep, count in sorted(running.items(), key=lambda t : t[0]))
        waiting_str = ' '.join('{} ({})'.format(dep, count) for dep, count in sorted(waiting_counts.items(), key=lambda t : t[0]))
        queue_str   = 'transactions : {}, scheduled : {}, waiting : {}'.format(self.transaction_queue.qsize(), self.schedule_queue.qsize(), len(self.waiting))
        # self.log.log('STATUS {}s'.format(round(duration, 2)))
        # self.log.log('  RUNNING {}'.format(running_str))
        # self.log.log('  WAITING {}'.format(waiting_str))
