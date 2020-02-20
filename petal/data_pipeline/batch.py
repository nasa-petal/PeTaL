import pickle

class Batch:
    def __init__(self):
        self.ids = []
        self.length = 0

    def __len__(self):
        return self.length

    def add(self, uuid):
        self.ids.append(uuid)
        self.length += 1

    def save(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self.ids, outfile)

    def load(self, filename):
        with open(filename, 'rb') as infile:
            self.ids = pickle.load(infile)

    def clear(self):
        del self.ids[:]
        self.length = 0
