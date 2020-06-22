import os
from datetime import datetime

LOG_DIR = 'data/logs/'

def make_directory(basename, directory=None):
    try:
        if not os.path.isdir(basename):
            os.mkdir(basename)
    except FileExistsError:
        pass
    if directory is not None:
        directory = basename + directory + '/'
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
        return directory
    else:
        return basename

def get_path(basename, name, directory=None, ending='.log'):
    time = datetime.now()

    path = make_directory(basename, directory=directory) + name
    path += '_' + time.strftime('%a_%d_%b_%y_%I_%M_%p') + ending

    base = path
    i = 1
    while os.path.isfile(base):
        base = path + '_' + str(i)
        i += 1
    path = base
    with open(path, 'w') as outfile:
        outfile.write('')
    return path

class Log:
    def __init__(self, name, directory=None):
        self.path = get_path(LOG_DIR, name, directory=directory)
        self.name = name

    def log(self, *messages, end='\n'):
        with open(self.path, 'a', encoding='utf-8') as outfile:
            for message in messages:
                outfile.write(str(message))
            outfile.write(end)
