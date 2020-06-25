import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def silence():
    '''
    Effectively make someone else's annoying code shut up, efficiently. Very useful.
    '''
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
